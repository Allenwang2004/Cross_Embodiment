"""optim.py -- the optimisers of the latent searches, as ask / tell objects (OOP refactor step 6, docs/refactor/PLAN.md).

Single latent (scripts/single_z_search.py, robot_z_search.py; z on Motivo's sphere |z| = 16):
  ES             perturbations eps ~ N(0, I) (independent: 2 * pairs of them, LSO's leave-one-out baseline; antithetic: pairs of
                 them and their negatives), centred ranks (or the raw costs) of the 2 * pairs candidates, Adam on z (or on the
                 coefficients of a --subspace), z re-projected after every step
  CEM            N(mean, std^2) per dimension on the sphere, then mean and std of the elites (std floor sigma_min)
  CMAES          pycma (popsize 2 * pairs, sigma0, seed + 1), candidates projected onto the sphere and told back, mean re-projected
  RandomSearch   the best latent so far + sigma * noise, on the sphere; moves only when a candidate beats it
Latent sequences (scripts/lso_search.py, robot_lso_search.py; one unit-sphere latent per control step, the policy gets 16 * z):
  LSO                   BFMTrack's LSO: coloured noise over time (colored_noise: sr then si), discounted return-to-go, optional
                        per-step ranks, leave-one-out baseline (or antithetic pairs), Adam (betas 0.8, 0.99) as ascent
  TrajectoryAdaptation  BFM-Zero's dual-loop annealed white noise, softmax average of the samples by their whole-rollout cost
Shared: Adam (m, v, t; bias correction with t = the step count: gen + 1, or the resumed t).

Every object owns its RNG (one np.random.default_rng(seed), one draw per ask) and draws exactly what the inline code of the
scripts at fa6c5f9 drew, in the same order and shape (tests/test_optim.py runs that code against these classes); the arithmetic is
written operation for operation as it was ([z + s eps, z - s eps] then project; the leave-one-out baseline; Adam's
lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)). The single-latent optimisers' ask(best_z) gets the search loop's
best-so-far latent (RandomSearch samples around it; the others ignore it); tell(cand, cost, best_z) updates `z`, the current
iterate. The sequence optimisers' ask() returns the n + 1 slot batch (slot 0 = the mean sequence `mu`); tell(Z, r, costs) takes
the per-step rewards (n + 1, S) and the costs of every slot. state_dict() / load_state_dict(): ES's z, Adam and RNG state
(robot_z_search's --resume and state.npz).

Pure helpers (stay functions, PLAN.md): colored_noise (from lso_search.py), return_to_go; project_z / rank_normalize /
rank_normalize_steps / unit_sphere are crossembodiment.geometry's.
"""
from __future__ import annotations

import json

import numpy as np

from .geometry import project_z, rank_normalize, rank_normalize_steps, unit_sphere


# ------------------------------------------------------------------ pure helpers

def colored_noise(beta, shape, rng):
    """Gaussian noise with power spectrum 1/f^beta along the LAST axis, unit variance (Timmer & Koenig 1995, as in
    the colorednoise package that BFMTrack / Eberhard et al. use). beta 0 = white. (scripts/lso_search.py's; draws the real
    parts sr, then the imaginary parts si)"""
    T = shape[-1]; f = np.fft.rfftfreq(T)
    s = np.maximum(f, 1.0 / T) ** (-beta / 2.0)
    w = s[1:].copy(); w[-1] *= (1 + (T % 2)) / 2.0
    sd = 2 * np.sqrt(np.sum(w ** 2)) / T
    sr = rng.standard_normal(shape[:-1] + (len(f),)) * s
    si = rng.standard_normal(shape[:-1] + (len(f),)) * s
    if T % 2 == 0:
        si[..., -1] = 0; sr[..., -1] *= np.sqrt(2)
    si[..., 0] = 0; sr[..., 0] *= np.sqrt(2)
    return np.fft.irfft(sr + 1j * si, n=T, axis=-1) / sd


def return_to_go(r, disc):
    """discounted return-to-go of every row of r (n, S): R_t = sum_j>=t disc[j - t] r_j (disc = gamma ** arange(S)), the
    lso_search.py / robot_lso_search.py expression np.convolve(x[::-1], disc)[:S][::-1] per row"""
    S = len(disc)
    return np.stack([np.convolve(x[::-1], disc)[:S][::-1] for x in r])


def with_mean(mu, samples):
    """the LSO / TA batch: slot 0 = the mean sequence, then the samples (BatchRollout.with_mean)"""
    return np.concatenate([mu[None], samples])


# ------------------------------------------------------------------ Adam

class Adam:
    """Adam's moments for one parameter array. step(g) -> lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps) with t the
    step count after this step (t starts at `t`: 0 for a new search, the saved count for a resumed one). The caller applies
    the step: descent z - step (single latent), ascent mu + step (LSO)."""

    def __init__(self, lr, size, betas=(0.9, 0.999), eps=1e-8, t=0):
        self.lr, self.eps, self.t = lr, eps, t
        self.b1, self.b2 = betas
        self.m = np.zeros(size)
        self.v = np.zeros_like(self.m)

    def step(self, g):
        b1, b2 = self.b1, self.b2
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g * g
        self.t += 1
        return self.lr * (self.m / (1 - b1 ** self.t)) / (np.sqrt(self.v / (1 - b2 ** self.t)) + self.eps)


# ------------------------------------------------------------------ single latent

class ES:
    """ES on z (single_z_search.py --algo es, robot_z_search.py --algo es).

    z          start latent (float64, on the sphere); `z` is the current iterate
    sampling   independent (2 * pairs eps, leave-one-out baseline) | antithetic (pairs eps and -eps, pair differences)
    fitness    rank (centred ranks) | raw (the costs as float64)
    subspace   None, or (K, 256) orthonormal rows U: eps = xi @ U with xi = sqrt(256 / K) * N(0, I_K); Adam on the K coefficients
               and the step sqrt(256 / K) * step @ U (single_z_search.py --subspace)
    """
    name = "es"

    def __init__(self, z, sigma, lr, pairs, seed=0, sampling="independent", fitness="rank", subspace=None):
        self.z = z
        self.sigma, self.pairs, self.sampling, self.fitness = sigma, pairs, sampling, fitness
        self.rng = np.random.default_rng(seed)
        self.n_eps = pairs if sampling == "antithetic" else 2 * pairs
        self.U = subspace
        if subspace is not None:
            self.sub_scale = np.sqrt(z.size / len(subspace))
        # Adam lives in the search coordinates: z itself, or the subspace coefficients
        self.adam = Adam(lr, z.size if subspace is None else len(subspace))
        self._noise = None

    def draw(self):
        """this step's noise: eps (n_eps, 256), and what the gradient weights (eps, or xi in a subspace)"""
        if self.U is None:
            eps = self.rng.standard_normal((self.n_eps, self.z.size))
            self._noise = eps
        else:
            xi = self.sub_scale * self.rng.standard_normal((self.n_eps, len(self.U)))
            eps = xi @ self.U
            self._noise = xi
        return eps

    def ask(self, best_z=None):
        eps = self.draw()
        z, sigma = self.z, self.sigma
        if self.sampling == "antithetic":
            return project_z(np.concatenate([z + sigma * eps, z - sigma * eps], axis=0))
        return project_z(z + sigma * eps)

    def gradient(self, cost):
        s = rank_normalize(cost) if self.fitness == "rank" else np.asarray(cost, dtype=np.float64)
        E, p, sigma = self._noise, self.pairs, self.sigma
        if self.sampling == "antithetic":
            return ((s[:p] - s[p:])[:, None] * E).sum(0) / (2 * p * sigma)
        adv = s - (s.sum() - s) / (len(s) - 1)          # leave-one-out baseline, as lso_search.py
        return (adv[:, None] * E).sum(0) / (2 * p * sigma)

    def tell(self, cand, cost, best_z=None):
        step = self.adam.step(self.gradient(cost))
        self.z = project_z(self.z - (step if self.U is None else self.sub_scale * step @ self.U))

    @property
    def step_sigma(self):
        return self.sigma

    def state_dict(self):
        """z, Adam's m, v and step count t, the RNG state (robot_z_search.py's state.npz holds these as z, m, v, t_adam, rng)"""
        return dict(z=self.z, m=self.adam.m, v=self.adam.v, t=self.adam.t, rng=json.dumps(self.rng.bit_generator.state))

    def load_state_dict(self, S):
        """the inverse of state_dict(); rng: the JSON string of the bit generator's state (as state.npz stores it)"""
        self.z, self.adam.m, self.adam.v, self.adam.t = S["z"], S["m"], S["v"], int(S["t"])
        self.rng.bit_generator.state = json.loads(str(S["rng"]))

    def skip(self, steps):
        """advance the noise stream past `steps` earlier steps (a resume without state.npz: Adam restarts)"""
        for _ in range(steps):
            self.rng.standard_normal((self.n_eps, self.z.size))


class CEM:
    """cross-entropy method on the sphere: candidates project(z + std * N(0, I)), then z = project(mean of the `elites` best),
    std = max(std of the elites, sigma_min); std starts at sigma everywhere, sigma_min defaults to sigma / 5."""
    name = "cem"

    def __init__(self, z, sigma, n, seed=0, elites=4, sigma_min=None):
        self.z, self.n, self.elites = z, n, elites
        self.rng = np.random.default_rng(seed)
        self.std = np.full(z.size, float(sigma))
        self.sigma_min = sigma_min if sigma_min is not None else sigma / 5

    def ask(self, best_z=None):
        return project_z(self.z + self.std * self.rng.standard_normal((self.n, self.z.size)))

    def tell(self, cand, cost, best_z=None):
        elite = cand[np.argsort(cost)[:self.elites]]
        self.z = project_z(elite.mean(0)); self.std = np.maximum(elite.std(0), self.sigma_min)

    @property
    def step_sigma(self):
        return float(self.std.mean())


class CMAES:
    """pycma's CMA-ES: popsize n, sigma0 sigma, seed + 1 (pycma's own RNG); candidates projected onto the sphere, the
    projected candidates told back, the mean kept on the sphere so sigma stays a fractional perturbation."""
    name = "cmaes"

    def __init__(self, z, sigma, n, seed=0):
        import cma
        self.z = z
        self.es = cma.CMAEvolutionStrategy(z.copy(), sigma, {"popsize": n, "seed": seed + 1, "verbose": -9})

    def ask(self, best_z=None):
        return project_z(np.asarray(self.es.ask(), dtype=np.float64))

    def tell(self, cand, cost, best_z=None):
        self.es.tell(list(cand), list(map(float, cost)))
        self.es.mean = project_z(np.asarray(self.es.mean, dtype=np.float64))
        self.z = self.es.mean.copy()

    @property
    def step_sigma(self):
        return float(self.es.sigma)


class RandomSearch:
    """candidates = project(best_z + sigma * N(0, I)) around the search's best latent so far; z follows the best."""
    name = "random"

    def __init__(self, z, sigma, n, seed=0):
        self.z, self.sigma, self.n = z, sigma, n
        self.rng = np.random.default_rng(seed)

    def ask(self, best_z=None):
        return project_z(best_z + self.sigma * self.rng.standard_normal((self.n, best_z.size)))

    def tell(self, cand, cost, best_z=None):
        self.z = best_z.copy()

    @property
    def step_sigma(self):
        return self.sigma


def single_latent_optimizer(algo, z, sigma, lr, pairs, seed, sampling="independent", fitness="rank", subspace=None,
                            cem_elites=4, cem_sigma_min=None):
    """--algo es | cem | random | cmaes of single_z_search.py / robot_z_search.py (2 * pairs candidates per step)"""
    n = 2 * pairs
    if algo == "es":
        return ES(z, sigma, lr, pairs, seed=seed, sampling=sampling, fitness=fitness, subspace=subspace)
    if algo == "cem":
        return CEM(z, sigma, n, seed=seed, elites=cem_elites, sigma_min=cem_sigma_min)
    if algo == "random":
        return RandomSearch(z, sigma, n, seed=seed)
    if algo == "cmaes":
        return CMAES(z, sigma, n, seed=seed)
    raise ValueError(f"unknown --algo {algo}")


# ------------------------------------------------------------------ latent sequences

class LSO:
    """BFMTrack's latent sequence optimisation. mu (S, 256) on the unit sphere; per ask n sequences
    unit(mu + sigma * eps), eps = colored_noise(beta, (n, 256, S)) transposed to (n, S, 256) (antithetic: n / 2 of them and
    their negatives); per tell the returns-to-go R of the samples' per-step rewards (slots 1..n), optional per-step centred
    ranks, the leave-one-out advantage (antithetic: the pair differences), g = mean_n A eps / sigma, mu = unit(mu + Adam(g))."""
    name = "lso"

    def __init__(self, mu, sigma, lr, n, seed=0, beta=1.0, gamma=0.97, betas=(0.8, 0.99), rank=False, antithetic=False):
        self.mu, self.sigma, self.n, self.beta = mu, sigma, n, beta
        self.rank, self.antithetic = rank, antithetic
        self.rng = np.random.default_rng(seed)
        self.disc = gamma ** np.arange(mu.shape[0])
        self.adam = Adam(lr, mu.shape, betas=betas, eps=1e-8)
        self.eps = None

    def ask(self):
        S = self.mu.shape[0]
        if self.antithetic:                                                    # pairs: eps^i, -eps^i
            h = colored_noise(self.beta, (self.n // 2, 256, S), self.rng).transpose(0, 2, 1)
            self.eps = np.concatenate([h, -h])
        else:
            self.eps = colored_noise(self.beta, (self.n, 256, S), self.rng).transpose(0, 2, 1)   # (n, S, 256)
        return with_mean(self.mu, unit_sphere(self.mu[None] + self.sigma * self.eps))

    def tell(self, Z, r, costs):
        R = return_to_go(r[1:], self.disc)                                     # discounted return-to-go of the samples
        if self.rank:                                                          # centred ranks, per step
            R = rank_normalize_steps(R)
        eps = self.eps
        if self.antithetic:
            k = self.n // 2
            g = ((R[:k] - R[k:])[:, :, None] * eps[:k]).sum(0) / (2 * k * self.sigma)
        else:
            A = R - (R.sum(0, keepdims=True) - R) / (self.n - 1)
            g = (A[:, :, None] * eps).mean(0) / self.sigma
        self.mu = unit_sphere(self.mu + self.adam.step(g))


class TrajectoryAdaptation:
    """BFM-Zero's trajectory adaptation (dual-loop annealing as DIAL-MPC): per ask n sequences unit(mu + sig * eps), white
    eps (n, S, 256), sig = sigma * exp(-(M - (M - it)) / (beta1 M)) * exp(-(S - h) / (beta2 S)) for h = 1..S (it = the
    iteration, M = iters); per tell mu = unit(softmax-weighted mean of the samples), weights exp(-(zc - min zc) / temp) of the
    samples' costs z-scored."""
    name = "ta"

    def __init__(self, mu, sigma, n, iters, seed=0, beta1=0.85, beta2=0.9, temp=0.05):
        self.mu, self.sigma, self.n, self.iters = mu, sigma, n, iters
        self.beta1, self.temp = beta1, temp
        self.rng = np.random.default_rng(seed)
        S = mu.shape[0]
        self.ta_h = np.exp(-(S - np.arange(1, S + 1)) / (beta2 * S))           # horizon factor, 1 at the last step
        self.it, self.w = 0, None

    def ask(self):
        it, M = self.it, self.iters
        sig = self.sigma * np.exp(-(M - (M - it)) / (self.beta1 * M)) * self.ta_h   # i = M - it counts down from M
        eps = self.rng.standard_normal((self.n, self.mu.shape[0], 256))
        return with_mean(self.mu, unit_sphere(self.mu[None] + sig[None, :, None] * eps))

    def tell(self, Z, r, costs):
        c = costs[1:]; zc = (c - c.mean()) / (c.std() + 1e-12); w = np.exp(-(zc - zc.min()) / self.temp); w /= w.sum()
        self.mu = unit_sphere((w[:, None, None] * Z[1:]).sum(0))
        self.w = w; self.it += 1
