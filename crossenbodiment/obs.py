"""obs.py -- what the frozen actor (and B) sees of a simulated body: the ObsMapper strategies (OOP refactor step 5, docs/refactor/PLAN.md).

A mapper turns the simulator's output into the actor's (n, 358) observation, at the start of a rollout and after every step;
crossenbodiment/rollout.py BatchRollout records that mapped observation as the rollout's `obs` (what B scores).

  RawObs         the simulator's own proprio (humenv obs["proprio"]; a robot's SMPL-equivalent observation from RobotVecEnv)
  MultiplierObs  proprio x a fixed (358,) multiplier (model/obs_scale.build_obs_multiplier: per-body length ratios, or one
                 uniform ratio) -- single_z_search --obs-scale auto / <r>, rank_initial_cost, test_exact_obs's "mult"
  ExactObs       the ADULT-equivalent observation of the state by reverse retargeting (model/exact_obs.py ExactObs, root
                 scale s). in_env=True (every search): each env worker computes its own slot's observation right after its
                 physics step (model/exact_obs.ExactObsWrapper -> info["exact_proprio"]); the main-process ExactObs
                 (`.exact`, made by build()) is used for the start pose only. in_env=False (rollout_z_trace via
                 single_z_search.rollout, test_exact_obs): the main process computes every slot in turn -- same numbers,
                 slower. Not to be confused with model.exact_obs.ExactObs, the function object this strategy wraps.

Factories (the scripts' own selection rules, kept as they are):
  single_z_obs(...)   single_z_search.py --obs-scale auto | none | <r> | exact, including its root-scale rule for --body
                      other than "child": s = bodies.clip_ratio(reference, the adult's clip) (the registry's rule, the same
                      expression the script had); the child keeps ExactObs's default 0.611036
  lso_scale_kw(scale) lso_search.py --scale AS IT IS: None -> ExactObs's default 0.611036, the CHILD's s, for ANY body
  registry_scale(body, clip)  the body's s from the registry (crossenbodiment/bodies.py), for drivers that want it explicit

Every computation is the one the old loops did, in the same dtype: proprio * mul in the dtypes of the two arrays (float64), the
exact observation as model/exact_obs.py returns it (float32), np.repeat of the start observation over the slots (or one per
slot for per-slot start poses). The ExactObs observation of a state is a pure function of (qpos, qvel) (mj_resetData +
mj_forward + get_obs on its own one-env humenv), so computing it again per rollout gives the identical bits.
"""
from __future__ import annotations

import functools

import numpy as np

from . import paths

CHILD_EXACT_S = 0.611036        # model/exact_obs.ExactObs's default scale (the child's s; configs/bodies.yaml child.s)


class ObsMapper:
    """strategy interface. wrappers(): humenv env wrappers the mapping needs inside the workers (before make_humenv);
    start(first, init_qpos, init_qvel, n, per_slot): the observation before the first step (first = the simulator's
    SimStep after the reset; its proprio is None when the start observation was not fetched); step(out): after a step."""
    name = "?"

    def wrappers(self):
        return []

    def start(self, first, init_qpos, init_qvel, n, per_slot=False):
        raise NotImplementedError

    def step(self, out):
        raise NotImplementedError


class RawObs(ObsMapper):
    """the simulator's proprio as it is (no copy, as the loops had it: every step's array is a fresh one)."""
    name = "raw"

    def start(self, first, init_qpos, init_qvel, n, per_slot=False):
        assert first.proprio is not None, "RawObs needs the start observation (fetch_start_obs=True)"
        return first.proprio

    def step(self, out):
        return out.proprio


class MultiplierObs(ObsMapper):
    """proprio * mul (model/obs_scale.py's (358,) multiplier)."""
    name = "multiplier"

    def __init__(self, mul):
        self.mul = mul

    def start(self, first, init_qpos, init_qvel, n, per_slot=False):
        assert first.proprio is not None, "MultiplierObs needs the start observation (fetch_start_obs=True)"
        return first.proprio * self.mul

    def step(self, out):
        return out.proprio * self.mul


class ExactObs(ObsMapper):
    """the adult-equivalent observation (model/exact_obs.py). xml: the simulated body; adult_xml: the adult the observation is
    computed on; scale: the root scale s (None = model/exact_obs's default 0.611036, passed to it as no argument at all, as the
    callers did). The paths are passed on as given: str() of them to the workers' ExactObsWrapper, as they are to the
    main-process ExactObs (single_z_search / lso_search's spellings)."""
    name = "exact"

    def __init__(self, xml=None, adult_xml=paths.ADULT_XML_REL, scale=None, in_env=True, exact=None):
        self.xml, self.adult_xml, self.in_env = xml, adult_xml, in_env
        self.kw = {} if scale is None else {"scale": scale}
        self.exact = exact                   # the main-process model.exact_obs.ExactObs (build(), or given)

    @classmethod
    def wrap(cls, exact, in_env=True):
        """a mapper around an existing model.exact_obs.ExactObs (single_z_search.rollout's and child_reward_eval.rollout's
        callers built theirs; with in_env the env's workers must already have the matching ExactObsWrapper)."""
        return cls(xml=None, exact=exact, in_env=in_env)

    @property
    def s(self):
        return self.exact.s

    def wrappers(self):
        if not self.in_env:
            return []
        from model.exact_obs import ExactObsWrapper
        return [functools.partial(ExactObsWrapper, child_xml=str(self.xml), adult_xml=str(self.adult_xml), **self.kw)]

    def build(self):
        """make the main-process model.exact_obs.ExactObs(xml, adult_xml[, scale]) (once) and return it."""
        if self.exact is None:
            from model.exact_obs import ExactObs as _ExactObs
            self.exact = _ExactObs(self.xml, self.adult_xml, **self.kw)
        return self.exact

    def start(self, first, init_qpos, init_qvel, n, per_slot=False):
        if init_qpos is None:
            raise SystemExit("--obs-scale exact needs --init reference")
        if per_slot:                         # test_exact_obs: every slot its own start pose
            return np.stack([self.exact(q, v) for q, v in zip(init_qpos, init_qvel)])
        return np.repeat(self.exact(init_qpos, init_qvel)[None], n, 0)

    def step(self, out):
        if self.in_env:
            return out.exact.copy()
        return np.stack([self.exact(q, v) for q, v in zip(out.qpos, out.qvel)])


def multiplier_or_raw(obs_mul):
    """single_z_search.rollout's obs_mul argument: None -> RawObs, else MultiplierObs(obs_mul)"""
    return RawObs() if obs_mul is None else MultiplierObs(obs_mul)


def single_z_obs(obs_scale, body, xml, ref_path=None, adult_clip=None, ref_xml=paths.ADULT_XML, parts="length"):
    """single_z_search.py's --obs-scale selection, rule for rule:
      exact        ExactObs(xml, ref_xml): --body child keeps ExactObs's default s (0.611036, no scale argument); any other
                   body reads s off the clip, bodies.clip_ratio(ref_path, adult_clip) (median ratio of the reference's root
                   x, y to the adult clip's: the expression the script had, now the registry's)
      none         RawObs
      auto / <r>   MultiplierObs(build_obs_multiplier(xml, ref_xml, mode=obs_scale, parts=parts))
    Returns the mapper; for exact, `mapper.kw` holds the scale (empty for the child) and build() makes the ExactObs."""
    if obs_scale == "exact":
        if body != "child":
            from .bodies import clip_ratio
            return ExactObs(xml, ref_xml, scale=clip_ratio(ref_path, adult_clip))
        return ExactObs(xml, ref_xml)
    from model.obs_scale import build_obs_multiplier
    return multiplier_or_raw(build_obs_multiplier(xml, ref_xml, mode=obs_scale, parts=parts, verbose=False))


def lso_scale_kw(scale):
    """lso_search.py --scale, AS IT IS.

    !!! WARNING: without --scale, lso_search.py gives ExactObs NO scale, i.e. model/exact_obs.py's default 0.611036 -- the
    !!! CHILD's root scale -- for ANY body. Every driver that runs lso_search.py on another body passes --scale explicitly
    !!! (body_52_test.py: 1 for the payload; amass_baselines.py: the tall body's s), and old commands rely on the default for
    !!! the child, so the default is kept for bit-identity. Drivers should pass registry_scale(body, clip) explicitly.
    """
    return {} if scale is None else {"scale": scale}


def registry_scale(body, clip=None):
    """the body's root scale s from the registry (crossenbodiment/bodies.py BODIES.scale): the child's 0.611036, a fixed
    body's literal, or single_z_search's rule on `clip`. Use it for lso_search.py --scale instead of relying on its default."""
    from .bodies import BODIES
    return BODIES.scale(body, clip)
