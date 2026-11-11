#!/usr/bin/env python3
"""run_queue.py -- run a commands file ("dir|cmd" per line, as the drivers' `commands` / `evalcmds` print it; "dir cmd" also read)
under the resource gate of crossenbodiment/queue.py (OOP refactor step 8): the replacement for the shell queues
(outputs/**/run_queue.sh, outputs/train_logs/queue/run_*.sh), which stay as they are for the experiments already running on them.

  uv run python scripts/run_queue.py outputs/tall_body_52/commands.txt --log outputs/tall_body_52/queue.log \\
      --sentinel outputs/tall_body_52/searches.done --max-jobs 8
  uv run python scripts/run_queue.py outputs/noanchor/evalcmds.txt --after outputs/noanchor/searches.done --done-file eval_qpos.npz
  uv run python scripts/run_queue.py outputs/noanchor/commands.txt --dry-run      # what would start now, and why; writes nothing

Per line: skipped when its folder has the done marker (--done-file, default summary.json; a --eval-save-qpos re-roll writes eval_qpos.npz last, after eval.csv), is
running under another run_queue.py (.queue.pid), or was started by an old shell queue (run.out without .rc; --retry-foreign runs it
again); otherwise `bash -c cmd` in its own session, cwd --cwd (default: the repository), the GPU-2 environment, output to
<dir>/run.out, the exit code to <dir>/.rc and the log (Taiwan time). A crashed run (exit != 0 or no marker) is retried --retries
times. Gate: at most --max-jobs of ours (or of the processes matching --count-pattern) at once; available memory minus the command's
estimate (--mem-gb, --mem-rule REGEX=GB; default 30 GB for robot / G1 searches, 12 GB otherwise) minus what our jobs started in the
last --ramp seconds need must stay >= --keep-gb (50); --cpu-idle-floor N (off by default); below --emergency-gb (20) available, no
more launches and the youngest of ours is killed (exit 2). --after FILE waits for a sentinel; --sentinel FILE is written when every
line is done (else FILE.failed). Exit: 0 all done, 1 some failed, 2 emergency stop, 3 a dependency failed.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from crossenbodiment.queue import MEM_DEFAULT, MEM_RULES, CommandQueue, ResourceGate, mem_estimate, read_commands  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("commands", help="the commands file: one 'dir|cmd' (or 'dir cmd') per line")
    p.add_argument("--log", default=None, help="queue log (default: queue.log next to the commands file)")
    p.add_argument("--cwd", default=str(REPO), help="where the commands run and their dirs are resolved (default: the repository)")
    p.add_argument("--max-jobs", type=int, default=8, help="at most this many running at once (default 8)")
    p.add_argument("--count-pattern", default=None, help="regex over the box's process command lines counted against --max-jobs "
                                                         "(e.g. '^uv run python scripts/single_z_search.py.*outputs/x/')")
    p.add_argument("--keep-gb", type=float, default=50.0, help="available memory to keep after a launch (default 50)")
    p.add_argument("--emergency-gb", type=float, default=20.0, help="below this available: stop launching, kill our youngest (20)")
    p.add_argument("--mem-gb", type=float, default=MEM_DEFAULT, help=f"memory estimate of a command no rule matches ({MEM_DEFAULT:g})")
    p.add_argument("--mem-rule", action="append", default=[], metavar="REGEX=GB",
                   help="estimate for commands matching REGEX (repeatable; first match wins; before the built-in robot rule)")
    p.add_argument("--ramp", type=float, default=120.0, help="seconds a new job counts with its full estimate (default 120)")
    p.add_argument("--cpu-idle-floor", type=float, default=None, help="launch only while CPU idle >= this %% (default: not checked)")
    p.add_argument("--done-file", default="summary.json", help="a line is done when <dir>/<this> exists (default summary.json)")
    p.add_argument("--retries", type=int, default=1, help="re-runs of a crashed line (default 1)")
    p.add_argument("--retry-foreign", action="store_true", help="also run lines whose run.out came from an old shell queue")
    p.add_argument("--after", action="append", default=[], help="wait for this file before the first launch (repeatable)")
    p.add_argument("--sentinel", default=None, help="written when every line is done (else <sentinel>.failed)")
    p.add_argument("--stagger", type=float, default=30.0, help="seconds between two launches (default 30)")
    p.add_argument("--poll", type=float, default=10.0, help="seconds between checks (default 10)")
    p.add_argument("--no-gpu-env", action="store_true", help="do not set the GPU-2 environment for the commands")
    p.add_argument("--dry-run", action="store_true", help="print what would start now and why; write nothing")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    rules = tuple((r.rsplit("=", 1)[0], float(r.rsplit("=", 1)[1])) for r in a.mem_rule) + MEM_RULES
    jobs = read_commands(a.commands, mem=lambda c: mem_estimate(c, rules, a.mem_gb))
    gate = ResourceGate(max_jobs=a.max_jobs, keep_gb=a.keep_gb, emergency_gb=a.emergency_gb, cpu_idle_floor=a.cpu_idle_floor,
                        ramp_s=a.ramp, count_pattern=a.count_pattern)
    q = CommandQueue(jobs, gate, log=a.log or Path(a.commands).with_name("queue.log"), cwd=a.cwd, env={} if a.no_gpu_env else None,
                     done_file=a.done_file, retries=a.retries, retry_foreign=a.retry_foreign, after=a.after, sentinel=a.sentinel,
                     poll_s=a.poll, stagger_s=a.stagger)
    if a.dry_run:
        q.dry_run()
        return 0
    return q.run()


if __name__ == "__main__":
    sys.exit(main())
