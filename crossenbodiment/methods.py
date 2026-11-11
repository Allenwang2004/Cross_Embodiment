"""methods.py -- the methods of the baseline comparison as data: MethodSpec, METHODS, and the run-directory key each experiment used.

OOP refactor step 2 (docs/refactor/PLAN.md); step 11 makes these keys the code's one name per method. On disk nothing is renamed:
run folders keep their run-directory keys (MethodSpec.run_key, and per experiment RUNS), result files their keys (RESULT_KEYS).
The same run-directory key means different methods in different experiments, which is what this table untangles:
  "cem"   = CEM with OUR cost (cem_ours) in outputs/child_52_rerun, but CEM with PHC's reward alone (cem_phc, BFM-Zero's) in
            outputs/child_52_cem_phc, body_52/<body>, g1_52, g1_small and tall_body_52
  "lsor"  = LSO + ranks with BFMTrack's own cost (lsor_bfm) in outputs/child_52_lso, but + 0.05 heading + 0.005 position
            (lsor_w005) in body_52/<body>, g1_52, g1_small and amass_baselines/<body>; child_52_lso calls that one "lsor_w005"
  "es"    = LOES with the 0.3 anchor (loes_anchor, the Task page's runs before 11/11) everywhere but outputs/noanchor/<body> and
            outputs/tall_body_52, where it is LOES without the anchor (loes, the method since 11/11)
  "z0"    = the frozen latent (frozen): no run of its own, its rollout is origin_z.npz of a single-latent run
Per method: the search script (human bodies / robots), the method's own CLI flags (on top of the shared settings in COMMON), and
the summary.json values that identify it (what the drivers' analyse steps check). tests/test_registry.py reads one existing
run per (experiment, key) and checks these values.

    from crossenbodiment.methods import METHODS, method_of, key_in, experiment_keys
    METHODS["cem_phc"].flags                          -> "--objective phc --algo cem"
    METHODS["cem_phc"].run_key                        -> "cem" (the --algo value, and the run folders' key)
    method_of("outputs/child_52_rerun", "cem").key    -> "cem_ours"
    key_in("outputs/child_52_lso", "lsor_w005")       -> "lsor_w005";  key_in("outputs/g1_52", "lsor_w005") -> "lsor"
    experiment_keys("outputs/tall_body_52")           -> {"loes": "es", "cem_phc": "cem"}
    canonical("z0") -> "frozen"; canonical("cem") raises (ambiguous: use method_of)
"""
from __future__ import annotations

import types
from dataclasses import dataclass

SINGLE_SCRIPT, SEQUENCE_SCRIPT = "scripts/single_z_search.py", "scripts/lso_search.py"
ROBOT_SINGLE_SCRIPT, ROBOT_SEQUENCE_SCRIPT = "scripts/robot_z_search.py", "scripts/robot_lso_search.py"


def _ro(d=(), **kw):
    """a read-only dict"""
    return types.MappingProxyType({**dict(d), **kw})


# the settings every method of a kind shares on the Task page (body_52_test.py zcmd / lcmd, g1_small_test.py, child_52_lso.py);
# the budget (--evals / --iters, --budgets), --seed and --out vary per experiment and are not listed
COMMON = _ro(
    single=_ro(flags="--pairs 8 --sigma 0.05 --lr 0.03 --obs-scale exact --eval-every 0 --init-vel-scale 1",
               settings=_ro(pairs=8, sigma=0.05, lr=0.03, obs_scale="exact", init_vel_scale=1.0, sampling="independent",
                            frame_offset=1, pos_unit="adult")),
    sequence=_ro(flags="--n 16 --init-vel-scale 1 --trace budgets",
                 settings=_ro(n=16, sigma=0.0125, beta=1.0, antithetic=False, init_vel_scale=1.0, pos_unit="adult")),
    robot_single=_ro(flags="--fill ridge --gains assets/robots/g1/gains_perjoint.json --pairs 8 --sigma 0.05 --lr 0.03 "
                           "--sampling independent --init-vel reference --eval-every 0",
                     settings=_ro(fill="ridge", init_vel="reference", pairs=8, sigma=0.05, lr=0.03, sampling="independent")),
    robot_sequence=_ro(flags="--fill ridge --gains assets/robots/g1/gains_perjoint.json --n 16 --init-vel reference --trace budgets",
                       settings=_ro(fill="ridge", init_vel="reference", n=16, sigma=0.0125)),
)


@dataclass(frozen=True)
class MethodSpec:
    key: str                          # the canonical name (code, tables, figures); run folders keep run_key / RUNS
    display: str                      # the name in tables and figures
    kind: str                         # single (one latent) | sequence (a latent per step)
    script: str | None                # the search on the human bodies (None: not a search of its own)
    flags: str | None                 # the method's own flags for `script` ({clip}: the task clip); None: not run there
    settings: types.MappingProxyType  # summary.json values of `script` that identify the method
    robot_script: str | None = None
    robot_flags: str | None = None
    robot_settings: types.MappingProxyType | None = None
    source: str | None = None         # for native / frozen: where the rollout comes from
    note: str = ""
    run_key: str | None = None        # the run-directory key the current drivers give its runs ("<stem>_<run_key>_s<seed>":
                                      # the --algo / --method value); None: no run of its own. Per experiment: RUNS / key_in


_OURS = "--objective bfm --heading-weight 1.0 --pos-weight 0.1 --anchor data/origin_z/{clip}.npy"
_OURS_S = dict(objective="bfm", heading_weight=1.0, pos_weight=0.1)
_ROBOT_OURS = "--heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0.3 --ik-weight 1.0"
_ROBOT_OURS_S = dict(objective="bfm", heading_weight=1.0, pos_weight=0.1, anchor_weight=0.3, ik_weight=1.0)

_SPECS = (
    MethodSpec("loes", "LOES (ours)", "single", SINGLE_SCRIPT, _OURS + " --anchor-weight 0",
               _ro(algo="es", anchor_weight=0.0, **_OURS_S),
               ROBOT_SINGLE_SCRIPT, "--algo es --heading-weight 1.0 --pos-weight 0.1 --anchor-weight 0 --ik-weight 1.0",
               _ro(algo="es", objective="bfm", heading_weight=1.0, pos_weight=0.1, anchor_weight=0.0, ik_weight=1.0),
               note="our method since 11/11 (the anchor term dropped; --anchor stays given so anchor_dist is logged): "
                    "BFM term + 1.0 heading + 0.1 position (adult-equivalent metres)", run_key="es"),
    MethodSpec("loes_anchor", "LOES (ours, + anchor)", "single", SINGLE_SCRIPT, _OURS + " --anchor-weight 0.3",
               _ro(algo="es", anchor_weight=0.3, **_OURS_S),
               ROBOT_SINGLE_SCRIPT, f"--algo es {_ROBOT_OURS}", _ro(algo="es", **_ROBOT_OURS_S),
               note="the Task page's LOES before 11/11: + 0.3 x (|z - z0| / 16)^2 (robots: + 1.0 IK residual)", run_key="es"),
    MethodSpec("cem_ours", "CEM (our cost)", "single", SINGLE_SCRIPT, "--algo cem " + _OURS + " --anchor-weight 0.3",
               _ro(algo="cem", anchor_weight=0.3, **_OURS_S),
               note="CEM with LOES's cost (Comparison A, outputs/child_52_rerun only)", run_key="cem"),
    MethodSpec("cem_phc", "CEM (BFM-Zero)", "single", SINGLE_SCRIPT, "--objective phc --algo cem",
               _ro(algo="cem", objective="phc", heading_weight=0.0, pos_weight=0.0, anchor_weight=0.0),
               ROBOT_SINGLE_SCRIPT, "--algo cem --objective phc --heading-weight 0 --pos-weight 0 --anchor-weight 0 --ik-weight 0",
               _ro(algo="cem", objective="phc", heading_weight=0.0, pos_weight=0.0, anchor_weight=0.0, ik_weight=0.0),
               note="BFM-Zero's tracking reward (model/phc_reward.py, lengths / the body's s) alone", run_key="cem"),
    MethodSpec("random", "Random Search", "single", SINGLE_SCRIPT, "--algo random " + _OURS + " --anchor-weight 0.3",
               _ro(algo="random", anchor_weight=0.3, **_OURS_S),
               ROBOT_SINGLE_SCRIPT, f"--algo random {_ROBOT_OURS}", _ro(algo="random", **_ROBOT_OURS_S),
               note="our cost with the anchor, as LOES's runs of the same experiment", run_key="random"),
    MethodSpec("cmaes", "CMA-ES", "single", SINGLE_SCRIPT, "--algo cmaes " + _OURS + " --anchor-weight 0.3",
               _ro(algo="cmaes", anchor_weight=0.3, **_OURS_S),
               ROBOT_SINGLE_SCRIPT, f"--algo cmaes {_ROBOT_OURS}", _ro(algo="cmaes", **_ROBOT_OURS_S),
               note="pycma, seed + 1; our cost with the anchor", run_key="cmaes"),
    MethodSpec("lso", "LSO (BFMTrack)", "sequence", SEQUENCE_SCRIPT, "",
               _ro(method="lso", objective="bfm", rank=False, heading_weight=0.0, pos_weight=0.0),
               ROBOT_SEQUENCE_SCRIPT, "--heading-weight 0 --pos-weight 0 --ik-weight 0",
               _ro(method="lso", objective="bfm", rank=False, heading_weight=0.0, pos_weight=0.0, ik_weight=0.0),
               note="BFMTrack's values and its own cost (the BFM term alone)", run_key="lso"),
    MethodSpec("lsor_bfm", "LSOR (BFMTrack cost)", "sequence", SEQUENCE_SCRIPT, "--rank",
               _ro(method="lso", objective="bfm", rank=True, heading_weight=0.0, pos_weight=0.0),
               note="LSO with centred ranks per step, BFMTrack's cost (outputs/child_52_lso 'lsor' only)", run_key="lsor"),
    MethodSpec("lsor_w005", "LSOR (ours)", "sequence", SEQUENCE_SCRIPT, "--rank --heading-weight 0.05 --pos-weight 0.005",
               _ro(method="lso", objective="bfm", rank=True, heading_weight=0.05, pos_weight=0.005),
               ROBOT_SEQUENCE_SCRIPT, "--rank --heading-weight 0.05 --pos-weight 0.005 --ik-weight 1.0",
               _ro(method="lso", objective="bfm", rank=True, heading_weight=0.05, pos_weight=0.005, ik_weight=1.0),
               note="ranks + our cost x 0.05 (the user's choice of 11/08, outputs/lsor_mpjae_test)", run_key="lsor"),
    MethodSpec("lsor_ours", "LSOR (our cost x 1)", "sequence", SEQUENCE_SCRIPT, "--rank --heading-weight 1.0 --pos-weight 0.1",
               _ro(method="lso", objective="bfm", rank=True, heading_weight=1.0, pos_weight=0.1),
               note="ranks + 1.0 heading + 0.1 position, no anchor (outputs/child_52_lso only)", run_key="lsor_ours"),
    MethodSpec("ta", "Trajectory Adaptation (BFM-Zero)", "sequence", SEQUENCE_SCRIPT, "--method ta --objective phc",
               _ro(method="ta", objective="phc", rank=False, heading_weight=0.0, pos_weight=0.0),
               ROBOT_SEQUENCE_SCRIPT, "--method ta --objective phc --heading-weight 0 --pos-weight 0 --ik-weight 0",
               _ro(method="ta", objective="phc", rank=False, heading_weight=0.0, pos_weight=0.0, ik_weight=0.0),
               note="from ER L = 8, PHC's tracking reward", run_key="ta"),
    MethodSpec("native", "Native BFM Tracking Inference", "sequence", None, None, _ro(),
               source="rollouts.npz q_ER8 of a latent-sequence run (Motivo's ER L = 8, the zero-shot rollout every run makes first)"),
    MethodSpec("frozen", "Frozen (z0)", "single", None, None, _ro(),
               source="origin_z.npz of a single-latent run (z0 re-projected onto the sphere, the rollout every search starts from)"),
)
METHODS = types.MappingProxyType({m.key: m for m in _SPECS})

# experiment folder -> (run-directory pattern, {run-directory key: method key}); <body> / <cond> stand for sub-folders.
# Only the folders the Task page and the current drivers read; nothing is renamed.
RUNS = types.MappingProxyType({
    "outputs/child_52_rerun": ("<stem>_<key>_s<seed>", _ro(es="loes_anchor", cem="cem_ours", random="random", cmaes="cmaes")),
    "outputs/child_52_cem_phc": ("<stem>_<key>_s<seed>", _ro(cem="cem_phc")),
    "outputs/child_52_lso": ("<stem>_<key>_s<seed>", _ro(lso="lso", lsor="lsor_bfm", lsor_w005="lsor_w005", lsor_ours="lsor_ours",
                                                          ta="ta")),   # + lso_ours: our cost without ranks, stopped at 24 of 260 runs (11/08)
    "outputs/body_52/<body>": ("<stem>_<key>_s0", _ro(es="loes_anchor", cem="cem_phc", random="random", cmaes="cmaes", lso="lso",
                                                       lsor="lsor_w005", ta="ta")),
    "outputs/g1_52": ("<stem>_<key>_s0", _ro(es="loes_anchor", cem="cem_phc", random="random", cmaes="cmaes", lso="lso",
                                             lsor="lsor_w005", ta="ta")),
    "outputs/g1_small": ("<stem>_<key>_s<seed>", _ro(es="loes_anchor", cem="cem_phc", random="random", cmaes="cmaes", lso="lso",
                                                     lsor="lsor_w005", ta="ta")),
    "outputs/noanchor/<body>": ("<stem>_<key>_s<seed>", _ro(es="loes")),
    "outputs/tall_body_52": ("<stem>_<key>_s0", _ro(es="loes", cem="cem_phc")),
    "outputs/amass_baselines/<body>": ("<clip>_<key>_s0", _ro(lso="lso", lsor="lsor_w005", ta="ta")),
    "outputs/walk_bodies": ("<body>_<key>_s0", _ro(es="loes_anchor")),
    "outputs/loes_lso_4096": ("<stem>_<key>_s<seed>", _ro(es="loes_anchor", lso="lso")),
    "outputs/kpkd/<body>_<cond>": ("<stem>_s<seed>", _ro({"": "loes_anchor"})),          # no method key in the folder name
})

# method keys inside result files (JSON / npz), where they differ from the run-directory keys
RESULT_KEYS = types.MappingProxyType({
    "outputs/fall_definitions": _ro(es="loes_anchor", cem="cem_ours", cem_phc="cem_phc", random="random", cmaes="cmaes"),
    "outputs/success_definitions": _ro(es="loes_anchor", cem_phc="cem_phc", random="random", cmaes="cmaes"),
    "outputs/body_52/<body>/results.json": _ro(es="loes_anchor", cem="cem_phc", random="random", cmaes="cmaes", lso="lso",
                                                lsor="lsor_w005", ta="ta", native="native", frozen="frozen"),
    "outputs/g1_52/results.json": _ro(es="loes_anchor", cem="cem_phc", random="random", cmaes="cmaes", lso="lso",
                                      lsor="lsor_w005", ta="ta", native="native", frozen="frozen"),
})
# with --loes-dir outputs/noanchor/child, fall_definitions.py / success_definitions.py / drift_success_child.py read "es" from the
# runs without the anchor (loes) and write *_noanchor.json


def method_of(experiment, key):
    """the MethodSpec behind run-directory `key` in `experiment` (a RUNS key, e.g. 'outputs/body_52/<body>')."""
    return METHODS[RUNS[experiment][1][key]]


def run_keys(method):
    """[(experiment, run-directory key)] of every folder in RUNS that holds runs of `method`."""
    return [(e, k) for e, (_, keys) in RUNS.items() for k, m in keys.items() if m == method]


# ------------------------------------------------------------------ step 11: one name per method in the code
# old names that mean one method wherever they appear ("z0": result files, palette, tables); the run-directory keys that mean
# different methods in different experiments need the experiment: method_of / key_in
ALIASES = types.MappingProxyType({"z0": "frozen"})
AMBIGUOUS = types.MappingProxyType({"es": ("loes", "loes_anchor"), "cem": ("cem_ours", "cem_phc"), "lsor": ("lsor_bfm", "lsor_w005")})


def canonical(name):
    """a method key or an unambiguous old name -> the canonical key ("z0" -> "frozen"); "es" / "cem" / "lsor" raise (which
    method they are depends on the experiment: method_of(experiment, key))"""
    if name in METHODS:
        return name
    if name in AMBIGUOUS:
        raise KeyError(f"{name!r} is ambiguous ({' or '.join(AMBIGUOUS[name])}); use method_of(experiment, {name!r})")
    return ALIASES[name]


def key_in(experiment, method):
    """the run-directory key of canonical `method` in `experiment` (a RUNS key): key_in("outputs/g1_52", "lsor_w005") -> "lsor"
    (the inverse of method_of)"""
    keys = [k for k, m in RUNS[experiment][1].items() if m == method]
    if len(keys) != 1:
        raise KeyError(f"{experiment}: {len(keys)} run-directory keys hold {method!r} ({keys})")
    return keys[0]


def experiment_keys(experiment):
    """{canonical method: run-directory key} of one experiment (RUNS), in RUNS' order: what a driver names its runs with"""
    return types.MappingProxyType({m: k for k, m in RUNS[experiment][1].items()})
