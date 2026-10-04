#!/usr/bin/env python3
"""wandb_clip_charts.py -- the trimmed eval charts, per-clip ones included, for a train_es run
that was started BEFORE train_es logged them (model/simple/train_es.py: eval_wandb_log).

The training process is not touched: this reads its ckpt dir's eval_history.json and writes
a companion wandb run "<name>-clips" in the same project, backfilling every eval so far and
then following the file until the training process (--pid) exits.

  uv run python scripts/wandb_clip_charts.py \
      --ckpt-dir outputs/simple_es/child_balanced/b500_global_anchor \
      --name child-bal-b500-global-anchor --of orhf0mqx --pid 171951
"""
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--name", required=True, help="the training run's wandb name")
    ap.add_argument("--of", default=None, help="the training run's wandb id, stored in the config")
    ap.add_argument("--pid", type=int, default=None, help="follow until this process exits")
    ap.add_argument("--project", default="crossenbodiment-simple")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    import wandb
    from model.simple.train_es import eval_wandb_log

    path = Path(a.ckpt_dir) / "eval_history.json"
    run = wandb.init(project=a.project, name=f"{a.name}-clips", job_type="eval-charts",
                     config={"companion_of": a.of, "ckpt_dir": a.ckpt_dir})
    done = 0
    while True:
        alive = a.pid is not None and os.path.exists(f"/proc/{a.pid}")
        try:
            H = [(e["update"], {k: v for k, v in e.items() if k != "update"})
                 for e in json.loads(path.read_text())]
        except json.JSONDecodeError:      # caught mid-rewrite; train_es rewrites it after every eval
            time.sleep(5)
            continue
        row = next((b for b in H[0][1] if b.endswith(":unseen")), None)
        clip_rows = {"train": row[: -len(":unseen")], "test": row} if row else None
        for i in range(done, len(H)):
            wandb.log(eval_wandb_log(H[: i + 1], clip_rows), step=H[i][0])
        if len(H) > done:
            print(f"[{time.strftime('%T')}] logged evals {done}..{len(H) - 1} (update {H[-1][0]})", flush=True)
        done = len(H)
        if not alive:
            break
        time.sleep(a.poll)
    run.finish()


if __name__ == "__main__":
    main()
