#!/usr/bin/env python3
"""strength_rollout.py -- what happens to the frozen policy's own motions when every actuator of the ORIGINAL adult
is weaker. Only the actuators change; same body, same z0, same start state (humenv's Default reset), so any change
in the rollout is the strength change alone.

  S1  <label>_gear:  every actuator's gear x --scale -> every joint torque x scale (PD stiffness, damping and the
                     torque ceiling all scale; the equilibrium angle of a command is unchanged)
  S2  <label>_cap:   every actuator's forcerange x --scale -> identical below the old ceiling x scale, only the
                     peak torque is capped earlier

The plain adult is run too, as the baseline under the same software (its distance from the recorded motion is
float-level on most clips, larger on chaotic ones). Per clip and body: the bfm cost against the adult's own rollout (1 - mean cos(B(s_t), B(g_t))), frames fallen,
end-point distance, mean joint-angle difference, and how often actuators sit at their torque ceiling.
Writes outputs/strength/<label>/summary.json and <variant>/<clip stem>.npz (qpos, for render_panels.py).
usage: uv run python scripts/strength_rollout.py [--scale 0.8 --label s08] [--with-cap]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model import bfm_align                 # noqa: E402
from model.bodies import up_z               # noqa: E402

CLIPS = ["crawl-0.4-0-d/crawl-0.4-0-d_7", "crawl-0.4-0-d/crawl-0.4-0-d_6", "headstand/headstand_9", "headstand/headstand_6",
         "jump-2/jump-2_8", "jump-2/jump-2_0", "move-ego-0-2-raisearms-h-h/move-ego-0-2-raisearms-h-h_6",
         "move-ego-0-2-raisearms-h-h/move-ego-0-2-raisearms-h-h_4", "move-ego-0-2-raisearms-m-h/move-ego-0-2-raisearms-m-h_7",
         "move-ego-0-2-raisearms-m-h/move-ego-0-2-raisearms-m-h_9", "raisearms-l-l/raisearms-l-l_8", "raisearms-l-l/raisearms-l-l_7",
         "rotate-x--5-0.8/rotate-x--5-0.8_6", "rotate-x--5-0.8/rotate-x--5-0.8_7", "rotate-z--5-0.8/rotate-z--5-0.8_0",
         "rotate-z--5-0.8/rotate-z--5-0.8_5", "move-ego-90-4/move-ego-90-4_9", "move-ego-0-2/move-ego-0-2_4"]
ADULT = REPO / "assets/robots/adult"


def build(label, scale, what):
    """assets/robots/<label>/robot.xml: the adult with every actuator's gear (what="gear") or forcerange ("cap") x scale
    (crossenbodiment/bodybuild ActuatorStrength, OOP refactor step 9: the same MjSpec edit, byte-identical robot.xml;
    parameter.json is the adult's keys + name / kind / parent / builder / edit, skeleton.json exported from the new XML)"""
    from crossenbodiment.bodybuild.builders import ActuatorStrength
    from crossenbodiment.bodybuild.skeleton import dump_skeleton
    b = ActuatorStrength(label, scale, what)
    b.build(log=lambda *_: None)
    dump_skeleton(b.xml, b.dir / "skeleton.json")
    return b.xml


def rollout(model, env, z, T):
    obs, _ = env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    zt = torch.as_tensor(np.asarray(z, np.float32).reshape(1, -1))
    q, o, sat = [], [], []
    for _ in range(T):
        with torch.no_grad():
            a = model.act(torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0), zt, mean=True)
        obs, _, _, _, info = env.step(a.numpy().ravel())
        q.append(info["qpos"].copy()); o.append(obs["proprio"].copy())
        sat.append(np.abs(d.actuator_force) >= 0.999 * m.actuator_forcerange[:, 1])
    return np.array(q), np.array(o, np.float32), np.array(sat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--label", default="s08")
    ap.add_argument("--with-cap", action="store_true", help="also run S2 (forcerange x scale)")
    a = ap.parse_args()
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    torch.set_num_threads(4)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to("cpu").eval()
    ap_bodies = {"adult": ADULT / "robot.xml", "S1_gear": build(f"adult_{a.label}_gear", a.scale, "gear")}
    if a.with_cap:
        ap_bodies["S2_cap"] = build(f"adult_{a.label}_cap", a.scale, "cap")
    bodies = ap_bodies
    envs = {k: make_humenv(num_envs=1, task=None, xml=str(x), state_init="Default")[0] for k, x in bodies.items()}
    out = REPO / "outputs/strength" / a.label
    res = {}
    for c in CLIPS:
        f = np.load(REPO / f"data/origin_action/{c}.npz"); ref = np.load(REPO / f"data/origin_motion/{c}.npz")["qpos"]
        T = len(ref)
        Bg = bfm_align.reference_embeddings(model, envs["adult"], ref, "cpu")
        d0 = ref[-1, :2] - ref[0, :2]; L0 = np.linalg.norm(d0)
        res[c] = {}
        for k, env in envs.items():
            q, o, sat = rollout(model, env, f["z"], T)
            # the plain adult is the baseline: float-level differences from the original recording stay ~1e-4 rad on
            # most clips but are amplified on chaotic ones (contacts), so its own row shows how far that goes
            u = np.array([up_z(x) for x in q[:, 3:7]]); ur = np.array([up_z(x) for x in ref[:, 3:7]])
            down = (u < ur - 0.8) | (q[:, 2] < 0.5 * ref[:, 2])
            cost = float(bfm_align.batch_bfm_align(model, o[None], Bg, "cpu")[0])
            res[c][k] = dict(cost=cost, down=int(down.sum()), end_xy=float(np.linalg.norm(q[-1, :2] - ref[-1, :2])),
                             travel=float(np.linalg.norm(q[-1, :2] - q[0, :2])), ref_travel=float(L0),
                             joint_deg=float(np.degrees(np.abs(q[:, 7:] - ref[:, 7:]).mean())),
                             pelvis=float(q[:, 2].mean()), ref_pelvis=float(ref[:, 2].mean()),
                             sat=float(sat.mean()), sat_steps=float(sat.any(1).mean()))
            p = out / k / f"{c.split('/')[1]}.npz"; p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, qpos=q, fps=30.0)
        r = res[c]
        print(f"{c.split('/')[1]:34s} " + " | ".join(
            f"{k}: cost {r[k]['cost']:.3f} down {r[k]['down']:3d} end {r[k]['end_xy']:5.2f} m joint {r[k]['joint_deg']:4.1f} sat {r[k]['sat_steps']:4.0%}"
            for k in envs), flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print("\nmedians over clips:")
    for k in envs:
        v = lambda key: np.median([res[c][k][key] for c in CLIPS])   # noqa: E731
        fell = sum(res[c][k]["down"] > 10 for c in CLIPS)
        print(f"  {k:8s} cost {v('cost'):.3f} | joint diff {v('joint_deg'):4.1f} deg | end-point off {v('end_xy'):5.2f} m | "
              f"clips with > 10 fallen frames {fell}/{len(CLIPS)} | steps with an actuator at its ceiling {v('sat_steps'):4.0%}")


if __name__ == "__main__":
    main()
