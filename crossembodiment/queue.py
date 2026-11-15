"""queue.py -- CommandQueue + ResourceGate: run an experiment's "dir|cmd" lines on the shared machine (OOP refactor step 8).

The ~25 shell queues (outputs/**/run_queue.sh, outputs/train_logs/queue/run_*.sh) each re-implemented the same loop: read
"dir|cmd" lines, wait while `pgrep -fc <pattern>` >= a cap or `free -g` available < 80 GB or CPU idle < 20 %, start
`bash -c "$cmd" > "$dir/run.out"` in the background, skip a folder that already has run.out, and chain stages by grepping
"all done" out of another queue's log. Their known faults: `echo "$(date ...) exit $?"` logged date's status (always 0), a run that
crashed was never retried (its run.out made every later queue skip it), and dependencies hung on log text. This module is the
one loop, in Python:

ResourceGate   may one more job start now?  (1) fewer than max_jobs of ours run (or, with count_pattern, fewer than max_jobs
               processes on the box match it, as the shell queues' pgrep counted every runner's jobs); (2) available memory
               (/proc/meminfo MemAvailable, = free -g's "available") minus the job's memory estimate, minus the estimates of our jobs
               started less than ramp_s ago (a search loads for 20-90 s before it uses its memory), stays >= keep_gb (default 50 GB,
               CLAUDE.md); (3) optional: CPU idle (/proc/stat over 2 s, as top -bn2 -d 2) >= cpu_idle_floor while one of ours runs
               (default off: the user said on 11/05 that CPU idle need not be checked).
               Emergency rule: available < emergency_gb (20 GB) -> no further launch, the youngest of OUR jobs is killed (its
               process group), logged, again every kill_grace_s while it stays low; the queue ends with exit code 2.
CommandQueue   the lines in file order. A line whose folder has the done marker (summary.json, or --done-file) is skipped; one whose
               .queue.pid names a live process is running elsewhere (skipped); one with run.out but neither marker nor .rc was
               started by an old shell queue (skipped like they did, unless retry_foreign); one whose last attempt crashed (our .rc,
               no marker) runs again. Each job: `bash -c cmd` in its own session, cwd the repository, the GPU-2 environment
               (CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1, TZ=Asia/Taipei),
               output to <dir>/run.out; on exit the child's real exit code goes to <dir>/.rc and the log. Success = exit 0 and the
               marker present; otherwise it is retried up to `retries` times (the old run.out kept as run.out.<attempt>).
               Dependencies: `after` files must exist before the first launch (a <file>.failed next to one ends the queue with
               exit 3); on completion the queue writes its `sentinel` (JSON counts) if every line is done, else <sentinel>.failed.
               Log lines are stamped in Taiwan time (Asia/Taipei). dry_run() prints what would launch now and why, writing nothing.
Exit codes of run(): 0 every line done, 1 some failed, 2 emergency stop, 3 a dependency failed.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import paths

TZ = ZoneInfo("Asia/Taipei")
GPU_ENV = dict(CUDA_VISIBLE_DEVICES="2", MUJOCO_EGL_DEVICE_ID="2", MUJOCO_GL="egl", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", TZ="Asia/Taipei")
MEM_RULES = ((r"robot_(z|lso)_search|g1_", 30.0),)          # G1 searches: ~17 GB measured, 30 GB budgeted (tests/golden/run.sh)
MEM_DEFAULT = 12.0                                           # a human-body search: ~9-10 GB (CLAUDE.md), 12 GB budgeted


def stamp():
    """now in Taiwan time, as the queues log it (MM-DD HH:MM:SS)"""
    return datetime.now(TZ).strftime("%m-%d %H:%M:%S")


def available_gb():
    """MemAvailable in GiB (what `free -g` prints as available, unrounded)"""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def _cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return v[3], sum(v[:8])                                   # idle, total (user nice system idle iowait irq softirq steal)


def cpu_idle(interval=2.0):
    """CPU idle share (%) over `interval` seconds (top -bn2 -d 2's second "%Cpu(s) ... id")"""
    i0, t0 = _cpu_times(); time.sleep(interval); i1, t1 = _cpu_times()
    return 100.0 * (i1 - i0) / max(t1 - t0, 1)


def count_processes(pattern, exclude=()):
    """processes whose command line (args joined by spaces, as pgrep -f sees it) matches the regex `pattern` (re.search), our
    own process and `exclude` pids left out"""
    rx, n, me = re.compile(pattern), 0, {os.getpid(), *exclude}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) in me:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        n += bool(cmd) and bool(rx.search(cmd))
    return n


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def mem_estimate(cmd, rules=MEM_RULES, default=MEM_DEFAULT):
    """GB a command is expected to need: the first rule whose regex matches the command, else `default`"""
    for rx, gb in rules:
        if re.search(rx, cmd):
            return float(gb)
    return float(default)


@dataclass
class Job:
    """one "dir|cmd" line"""
    dir: str
    cmd: str
    line: int = 0
    mem_gb: float = MEM_DEFAULT
    attempts: int = 0
    proc: subprocess.Popen | None = field(default=None, repr=False)
    started: float = 0.0


def parse_lines(lines, mem=mem_estimate):
    """["dir|cmd"] (or the xargs-era "dir cmd": split at the first space when there is no "|") -> [Job]; blank lines and
    lines starting with # are skipped"""
    jobs = []
    for n, l in enumerate(lines, 1):
        l = l.rstrip("\n")
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        d, sep, cmd = l.partition("|") if "|" in l else l.partition(" ")
        assert sep and d and cmd, f"line {n}: neither 'dir|cmd' nor 'dir cmd': {l!r}"
        jobs.append(Job(d, cmd, n, mem(cmd)))
    return jobs


def read_commands(path, mem=mem_estimate):
    with open(path) as f:
        return parse_lines(f, mem)


class ResourceGate:
    """may one more job start now? (module docstring)"""

    def __init__(self, max_jobs=8, keep_gb=50.0, emergency_gb=20.0, cpu_idle_floor=None, ramp_s=120.0, count_pattern=None,
                 avail=available_gb, idle=cpu_idle, count=count_processes, clock=time.monotonic):
        self.max_jobs, self.keep_gb, self.emergency_gb = max_jobs, keep_gb, emergency_gb
        self.cpu_idle_floor, self.ramp_s, self.count_pattern = cpu_idle_floor, ramp_s, count_pattern
        self.avail, self.idle, self.count, self.clock = avail, idle, count, clock

    def running(self, jobs):
        """how many count against the cap: ours, or the matching processes on the box if that is more"""
        n = len(jobs)
        if self.count_pattern:
            n = max(n, self.count(self.count_pattern))
        return n

    def check(self, job, jobs):
        """(ok, reason): may `job` start while `jobs` (ours) run?"""
        n = self.running(jobs)
        if n >= self.max_jobs:
            return False, f"{n} running (cap {self.max_jobs})"
        a = self.avail(); now = self.clock()
        ramping = sum(j.mem_gb for j in jobs if now - j.started < self.ramp_s)
        after = a - ramping - job.mem_gb
        if after < self.keep_gb:
            return False, (f"available {a:.0f} GB - {ramping:.0f} GB starting - {job.mem_gb:.0f} GB estimate = {after:.0f} GB "
                           f"< {self.keep_gb:.0f} GB")
        if self.cpu_idle_floor is not None and n >= 1:
            i = self.idle()
            if i < self.cpu_idle_floor:
                return False, f"CPU idle {i:.0f} % < {self.cpu_idle_floor:.0f} % with {n} running"
        return True, f"available {a:.0f} GB, {n} running"

    def emergency(self):
        """(low, available GB): is available memory below the emergency floor?"""
        a = self.avail()
        return a < self.emergency_gb, a


class CommandQueue:
    """run the lines of a commands file under a ResourceGate (module docstring)"""

    def __init__(self, jobs, gate=None, log=None, cwd=None, env=None, done_file="summary.json", retries=1, retry_foreign=False,
                 after=(), sentinel=None, poll_s=10.0, stagger_s=30.0, kill_grace_s=30.0, out_name="run.out", echo=True,
                 sleep=time.sleep, clock=time.monotonic):
        self.jobs, self.gate = list(jobs), gate or ResourceGate()
        self.cwd = Path(cwd or paths.REPO)
        self.log_path = Path(log) if log else None
        self.env = {**os.environ, **(GPU_ENV if env is None else env)}
        self.done_file, self.retries, self.retry_foreign = done_file, retries, retry_foreign
        self.after, self.sentinel = [Path(a) for a in after], Path(sentinel) if sentinel else None
        self.poll_s, self.stagger_s, self.kill_grace_s, self.out_name = poll_s, stagger_s, kill_grace_s, out_name
        self.echo, self.sleep, self.clock = echo, sleep, clock
        self.results = {}                                    # dir -> dict(rc, seconds, attempts, done)
        self.killed = []

    # ------------------------------------------------------------------ files of a job
    def path(self, job, name=""):
        return self.cwd / job.dir / name

    def done(self, job):
        return self.path(job, self.done_file).exists()

    def state(self, job):
        """done | running (a live .queue.pid) | foreign (run.out without .rc: an old shell queue's) | crashed (our .rc, no
        marker) | new"""
        if self.done(job):
            return "done"
        pf = self.path(job, ".queue.pid")
        if pf.exists():
            try:
                if pid_alive(int(pf.read_text().split()[0])):
                    return "running"
            except (ValueError, IndexError):
                pass
        if self.path(job, ".rc").exists():
            return "crashed"
        if self.path(job, self.out_name).exists():
            return "foreign"
        return "new"

    # ------------------------------------------------------------------ logging
    def log(self, msg):
        line = f"{stamp()} {msg}"
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    # ------------------------------------------------------------------ one job
    def launch(self, job):
        d = self.path(job); d.mkdir(parents=True, exist_ok=True)
        out = d / self.out_name
        if out.exists():                                     # an earlier attempt's output stays next to the new one
            k = 1
            while (d / f"{self.out_name}.{k}").exists():
                k += 1
            out.rename(d / f"{self.out_name}.{k}")
        rc_file = d / ".rc"
        if rc_file.exists():
            rc_file.unlink()
        job.attempts += 1
        with open(out, "w") as f:
            job.proc = subprocess.Popen(["bash", "-c", job.cmd], cwd=self.cwd, env=self.env, stdout=f, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        job.started = self.clock()
        (d / ".queue.pid").write_text(f"{job.proc.pid} {os.getpid()}\n")
        return job.proc

    def reap(self, job):
        """the job's process has ended: .rc, the log line, the result; returns (ok, rc)"""
        rc = job.proc.returncode; sec = self.clock() - job.started
        d = self.path(job)
        (d / ".rc").write_text(f"{rc}\n")
        pf = d / ".queue.pid"
        if pf.exists():
            pf.unlink()
        ok_marker = self.done(job); ok = rc == 0 and ok_marker
        note = "" if ok else (f" (no {self.done_file})" if rc == 0 else (f" ({self.done_file} present)" if ok_marker else ""))
        self.log(f"done {job.dir} exit {rc} ({sec:.0f} s, attempt {job.attempts}){note}")
        self.results[job.dir] = dict(rc=rc, seconds=round(sec, 1), attempts=job.attempts, done=ok_marker)
        return ok, rc

    def kill(self, job, why):
        try:
            os.killpg(job.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(50):
            if job.proc.poll() is not None:
                break
            time.sleep(0.2)
        else:
            try:
                os.killpg(job.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job.proc.wait()
        self.killed.append(job.dir)
        self.log(f"killed {job.dir} (pid {job.proc.pid}): {why}")

    # ------------------------------------------------------------------ the queue
    def wait_for_dependencies(self):
        """block until every `after` file exists; False if one has failed (<file>.failed)"""
        for a in self.after:
            told = False
            while not a.exists():
                if Path(str(a) + ".failed").exists():
                    self.log(f"dependency failed: {a}.failed exists; not starting")
                    return False
                if not told:
                    self.log(f"waiting for {a}"); told = True
                self.sleep(self.poll_s)
        return True

    def pending(self):
        """the jobs to run, with the reason for the others (logged)"""
        todo = []
        for j in self.jobs:
            s = self.state(j)
            if s in ("new", "crashed") or (s == "foreign" and self.retry_foreign):
                todo.append(j)
            else:
                self.results.setdefault(j.dir, dict(rc=None, seconds=None, attempts=0, done=s == "done", skipped=s))
        return todo

    def run(self):
        if not self.wait_for_dependencies():
            return 3
        todo = self.pending()
        skipped = {}
        for r in self.results.values():
            skipped[r["skipped"]] = skipped.get(r["skipped"], 0) + 1
        self.log(f"queue: {len(self.jobs)} lines, {len(todo)} to run, skipped {skipped or 'none'}")
        running, failed, stopped = [], [], False
        last_launch = last_kill = -1e18; last_reason = None
        while todo or running:
            for j in [j for j in running if j.proc.poll() is not None]:
                running.remove(j); ok, rc = self.reap(j)
                if ok:
                    continue
                crashed = not self.done(j)                   # exit != 0 or no marker; a marker with exit != 0 is not re-run
                if crashed and j.attempts <= self.retries and not stopped and j.dir not in self.killed:
                    self.log(f"retry {j.dir} (exit {rc}; attempt {j.attempts + 1} of {self.retries + 1})"); todo.append(j)
                else:
                    failed.append(j)
            low, a = self.gate.emergency()
            if low and running and self.clock() - last_kill >= self.kill_grace_s:
                y = max(running, key=lambda j: j.started)
                self.log(f"EMERGENCY: available {a:.1f} GB < {self.gate.emergency_gb:.0f} GB: no further launch; killing the youngest "
                         f"of ours, {y.dir}")
                self.kill(y, f"available {a:.1f} GB < {self.gate.emergency_gb:.0f} GB")
                last_kill = self.clock(); stopped = True
                continue
            if low and not stopped:
                self.log(f"EMERGENCY: available {a:.1f} GB < {self.gate.emergency_gb:.0f} GB: no further launch"); stopped = True
            if stopped:
                if not running:
                    break
                self.sleep(self.poll_s); continue
            if todo and self.clock() - last_launch >= self.stagger_s:
                j = todo[0]; ok, why = self.gate.check(j, running)
                if ok:
                    todo.pop(0); self.launch(j); running.append(j); last_launch = self.clock(); last_reason = None
                    self.log(f"start {j.dir} ({why}; estimate {j.mem_gb:.0f} GB; attempt {j.attempts})")
                    continue
                if why != last_reason:
                    self.log(f"waiting: {why}"); last_reason = why
            self.sleep(self.poll_s)
        return self.finish(failed, todo, stopped)

    def finish(self, failed, todo, stopped):
        n_done = sum(self.done(j) for j in self.jobs)
        summ = dict(lines=len(self.jobs), done=n_done, failed=[j.dir for j in failed], not_started=[j.dir for j in todo],
                    killed=self.killed, emergency_stop=stopped, finished=stamp())
        all_done = n_done == len(self.jobs)
        self.log(f"{'all done' if all_done else 'finished'}: {n_done} of {len(self.jobs)} done, {len(failed)} failed, "
                 f"{len(todo)} not started{', emergency stop' if stopped else ''}")
        if self.sentinel:
            target = self.sentinel if all_done else Path(str(self.sentinel) + ".failed")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(summ, indent=1) + "\n")
            self.log(f"wrote {target}")
        return 2 if stopped else (0 if all_done else 1)

    # ------------------------------------------------------------------ dry run
    def dry_run(self, out=sys.stdout):
        """print each line's state and what the gate would launch now (in file order, until the cap or the memory floor),
        writing nothing"""
        counts, would, simulated = {}, [], []
        for a in self.after:
            print(f"after {a}: {'present' if a.exists() else 'missing (the queue would wait)'}"
                  f"{' -- FAILED' if Path(str(a) + '.failed').exists() else ''}", file=out)
        for j in self.jobs:
            s = self.state(j); counts[s] = counts.get(s, 0) + 1
            runnable = s in ("new", "crashed") or (s == "foreign" and self.retry_foreign)
            if runnable:
                fake = Job(j.dir, j.cmd, j.line, j.mem_gb, started=self.clock())
                ok, why = self.gate.check(fake, simulated)
                if ok and len(would) < self.gate.max_jobs:
                    simulated.append(fake); would.append(j)
                    print(f"launch now  {j.dir}  ({j.mem_gb:.0f} GB; {why})", file=out)
                else:
                    print(f"later       {j.dir}  ({why})", file=out)
            else:
                print(f"skip {s:8s}{j.dir}", file=out)
        low, a = self.gate.emergency()
        print(f"\n{len(self.jobs)} lines: " + ", ".join(f"{k} {v}" for k, v in counts.items()) +
              f" | would launch {len(would)} now (cap {self.gate.max_jobs}, keep {self.gate.keep_gb:.0f} GB, available {a:.0f} GB"
              f"{', BELOW the emergency floor' if low else ''}) | done marker {self.done_file} | cwd {self.cwd}", file=out)
        return would
