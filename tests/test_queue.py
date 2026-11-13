"""Tests of OOP refactor step 8's queue runner (crossembodiment/queue.py, scripts/run_queue.py) with fake commands: sleep, exit
codes, retries, the memory-estimate gate, the job cap, dependencies by sentinel file, the emergency stop, old shell queues' folders,
and the dry run. Everything runs in outputs/refactor_tests/test_queue (inside the worktree) with `bash -c` one-liners; no GPU.

Run standalone (no pytest in the environment):  uv run python tests/test_queue.py      (~15 s)
"""
import io
import json
import os
import shutil
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from crossembodiment.queue import CommandQueue, ResourceGate, mem_estimate, parse_lines, stamp  # noqa: E402

WORK = REPO / "outputs/refactor_tests/test_queue"
FAST = dict(poll_s=0.05, stagger_s=0.0, echo=False, env={})


def fresh(name):
    d = WORK / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def gate(**kw):
    kw.setdefault("avail", lambda: 400.0)
    return ResourceGate(**kw)


def times(d, names):
    """(start, end) epoch seconds the fake commands wrote"""
    out = {}
    for n in names:
        f = d / n / "t"
        out[n] = tuple(float(x) for x in f.read_text().split()) if f.exists() else None
    return out


def timed(sec, marker=True):
    """a command that records its start and end and writes summary.json"""
    return (f"s=$(date +%s.%N); sleep {sec}; e=$(date +%s.%N); mkdir -p $D; echo $s $e > $D/t" +
            ("; echo {} > $D/summary.json" if marker else ""))


def lines(d, spec):
    """spec: [(name, cmd)] -> 'dir|cmd' lines with $D = the job's folder (absolute)"""
    return [f"{n}|D={d / n}; {c}" for n, c in spec]


class TestParse(unittest.TestCase):
    def test_formats(self):
        jobs = parse_lines(["outputs/x/a_s0|uv run python scripts/single_z_search.py --out outputs/x/a_s0\n", "\n", "# note\n",
                            "outputs/x/b_s0 uv run python scripts/robot_z_search.py --robot g1 --out outputs/x/b_s0\n"])
        self.assertEqual([j.dir for j in jobs], ["outputs/x/a_s0", "outputs/x/b_s0"])
        self.assertEqual(jobs[0].cmd, "uv run python scripts/single_z_search.py --out outputs/x/a_s0")
        self.assertEqual(jobs[1].cmd, "uv run python scripts/robot_z_search.py --robot g1 --out outputs/x/b_s0")
        self.assertEqual([j.line for j in jobs], [1, 4])
        self.assertEqual((jobs[0].mem_gb, jobs[1].mem_gb), (12.0, 30.0))       # the default estimates
        self.assertEqual(mem_estimate("x --eval-z z.npy", ((r"--eval-z", 5.0),)), 5.0)

    def test_stamp_is_taiwan_time(self):
        from datetime import datetime, timezone, timedelta
        want = datetime.now(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M")
        self.assertTrue(stamp().startswith(want[:8]))                          # date and hour in UTC+8


class TestRun(unittest.TestCase):
    def test_exit_codes_retry_skip_sentinel(self):
        d = fresh("codes")
        (d / "old").mkdir(); (d / "old" / "summary.json").write_text("{}")   # done before: skipped
        spec = [("ok", timed(0.1)), ("bad", "exit 7"), ("old", "echo should-not-run > $D/ran"), ("nomark", "true")]
        log = d / "queue.log"
        q = CommandQueue(parse_lines(lines(d, spec)), gate(), log=log, cwd=d, retries=1, sentinel=d / "all.done", **FAST)
        rc = q.run()
        self.assertEqual(rc, 1)                                                # two lines never got their marker
        self.assertEqual((d / "ok" / ".rc").read_text(), "0\n")
        self.assertEqual((d / "bad" / ".rc").read_text(), "7\n")              # the child's real exit code
        self.assertTrue((d / "bad" / "run.out.1").exists())                   # the first attempt's output kept
        self.assertFalse((d / "old" / "ran").exists())
        text = log.read_text()
        self.assertIn("done bad exit 7", text); self.assertIn("retry bad (exit 7; attempt 2 of 2)", text)
        self.assertIn("done nomark exit 0", text); self.assertIn("(no summary.json)", text)
        self.assertEqual(q.results["bad"]["attempts"], 2)
        self.assertFalse((d / "all.done").exists())
        s = json.loads((d / "all.done.failed").read_text())
        self.assertEqual(sorted(s["failed"]), ["bad", "nomark"]); self.assertEqual(s["done"], 2)
        self.assertFalse((d / "ok" / ".queue.pid").exists())

    def test_crashed_run_is_retried_on_the_next_queue(self):
        d = fresh("crashed")
        (d / "a").mkdir(); (d / "a" / "run.out").write_text("first attempt\n"); (d / "a" / ".rc").write_text("1\n")
        q = CommandQueue(parse_lines(lines(d, [("a", timed(0.05))])), gate(), cwd=d, **FAST)
        self.assertEqual(q.state(q.jobs[0]), "crashed")
        self.assertEqual(q.run(), 0)
        self.assertEqual((d / "a" / "run.out.1").read_text(), "first attempt\n")
        self.assertEqual((d / "a" / ".rc").read_text(), "0\n")

    def test_foreign_folder_skipped_unless_asked(self):
        d = fresh("foreign")
        (d / "a").mkdir(); (d / "a" / "run.out").write_text("an old shell queue's run\n")
        q = CommandQueue(parse_lines(lines(d, [("a", timed(0.05))])), gate(), cwd=d, **FAST)
        self.assertEqual(q.state(q.jobs[0]), "foreign")
        self.assertEqual(q.run(), 1); self.assertFalse((d / "a" / "summary.json").exists())
        q = CommandQueue(parse_lines(lines(d, [("a", timed(0.05))])), gate(), cwd=d, retry_foreign=True, **FAST)
        self.assertEqual(q.run(), 0); self.assertTrue((d / "a" / "summary.json").exists())

    def test_cap(self):
        d = fresh("cap")
        names = [f"j{i}" for i in range(5)]
        q = CommandQueue(parse_lines(lines(d, [(n, timed(0.4)) for n in names])), gate(max_jobs=2), cwd=d, **FAST)
        self.assertEqual(q.run(), 0)
        t = times(d, names)
        for x in sorted(v[0] for v in t.values()):                            # at every start, at most 2 running
            self.assertLessEqual(sum(s <= x + 1e-3 < e for s, e in t.values()), 2)

    def test_memory_estimate_gate(self):
        """70 GB available, keep 50: one 12 GB job fits (58), a second does not while the first is starting (70 - 12 - 12 =
        46 < 50), so the jobs run one after the other; a 5 GB job fits next to a starting 12 GB one (53)"""
        d = fresh("memory")
        q = CommandQueue(parse_lines(lines(d, [("a", timed(0.4)), ("b", timed(0.4))])), gate(avail=lambda: 70.0, ramp_s=60),
                         log=d / "queue.log", cwd=d, **FAST)
        self.assertEqual(q.run(), 0)
        t = times(d, ["a", "b"]); self.assertGreaterEqual(t["b"][0], t["a"][1] - 0.05)
        self.assertIn("waiting: available 70 GB - 12 GB starting - 12 GB estimate = 46 GB < 50 GB", (d / "queue.log").read_text())
        d2 = fresh("memory_small")
        jobs = parse_lines(lines(d2, [("a", timed(0.4)), ("b", timed(0.4))]), mem=lambda c: 12.0)
        jobs[1].mem_gb = 5.0
        q = CommandQueue(jobs, gate(avail=lambda: 70.0, ramp_s=60), cwd=d2, **FAST)
        self.assertEqual(q.run(), 0)
        t = times(d2, ["a", "b"]); self.assertLess(t["b"][0], t["a"][1])      # they overlapped

    def test_dependency_sentinel(self):
        d = fresh("deps")
        first = CommandQueue(parse_lines(lines(d, [("a", timed(0.5))])), gate(), cwd=d, sentinel=d / "a.done", **FAST)
        second = CommandQueue(parse_lines(lines(d, [("b", timed(0.05))])), gate(), cwd=d, after=[d / "a.done"], log=d / "b.log",
                              **FAST)
        th = threading.Thread(target=lambda: setattr(second, "rc", second.run())); th.start()
        time.sleep(0.2)
        self.assertFalse((d / "b").exists())                                  # still waiting for a.done
        self.assertEqual(first.run(), 0); th.join(10)
        self.assertEqual(second.rc, 0)
        self.assertGreaterEqual(times(d, ["b"])["b"][0], (d / "a.done").stat().st_mtime - 0.05)
        self.assertIn(f"waiting for {d / 'a.done'}", (d / "b.log").read_text())
        self.assertEqual(json.loads((d / "a.done").read_text())["done"], 1)
        (d / "x.done.failed").write_text("{}")                                # a failed dependency ends the queue
        q = CommandQueue(parse_lines(lines(d, [("c", timed(0.05))])), gate(), cwd=d, after=[d / "x.done"], **FAST)
        self.assertEqual(q.run(), 3); self.assertFalse((d / "c").exists())

    def test_emergency_stop(self):
        """available drops below 20 GB once the second job runs: no third launch, the youngest (the second) is killed and
        logged, the first finishes, exit 2"""
        d = fresh("emergency")
        low = d / "low"
        spec = [("a", timed(1.0)), ("b", f"touch {low}; sleep 30; echo {{}} > $D/summary.json"), ("c", timed(0.05))]
        g = gate(avail=lambda: 10.0 if low.exists() else 400.0)
        q = CommandQueue(parse_lines(lines(d, spec)), g, log=d / "queue.log", cwd=d, kill_grace_s=30, **dict(FAST, stagger_s=0.5))
        t0 = time.time(); rc = q.run()
        self.assertEqual(rc, 2); self.assertLess(time.time() - t0, 15)
        text = (d / "queue.log").read_text()
        self.assertIn("EMERGENCY: available 10.0 GB < 20 GB", text); self.assertIn("killed b", text)
        self.assertEqual(q.killed, ["b"])
        self.assertTrue((d / "a" / "summary.json").exists())                  # the older job was left to finish
        self.assertEqual((d / "b" / ".rc").read_text(), "-15\n")              # SIGTERM to its process group
        self.assertFalse((d / "c").exists())                                  # never launched


class TestDryRun(unittest.TestCase):
    def test_dry_run_writes_nothing(self):
        d = fresh("dry")
        (d / "done").mkdir(); (d / "done" / "summary.json").write_text("{}")
        (d / "foreign").mkdir(); (d / "foreign" / "run.out").write_text("")
        spec = [("done", "true"), ("foreign", "true"), ("g1", "uv run python scripts/robot_z_search.py --robot g1"),
                ("child", "uv run python scripts/single_z_search.py"), ("child2", "uv run python scripts/single_z_search.py")]
        before = sorted(p.relative_to(d) for p in d.rglob("*"))
        q = CommandQueue(parse_lines(lines(d, spec)), gate(avail=lambda: 100.0, max_jobs=8), log=d / "queue.log", cwd=d, **FAST)
        buf = io.StringIO()
        with redirect_stdout(buf):
            would = q.dry_run(out=sys.stdout)
        out = buf.getvalue()
        self.assertEqual(sorted(p.relative_to(d) for p in d.rglob("*")), before)   # nothing written, not even the log
        self.assertEqual([j.dir for j in would], ["g1", "child"])             # 100 - 30 = 70, 100 - 30 - 12 = 58: both >= 50
        self.assertIn("skip done    done", out); self.assertIn("skip foreign foreign", out)
        self.assertIn("later       child2  (available 100 GB - 42 GB starting - 12 GB estimate = 46 GB < 50 GB)", out)
        self.assertIn("launch now  g1", out); self.assertIn("would launch 2 now", out)


class TestCLI(unittest.TestCase):
    def test_run_queue_cli(self):
        """scripts/run_queue.py: a dry run prints and writes nothing; a real run of two lines writes .rc, the log and the sentinel"""
        import subprocess
        d = fresh("cli")
        (d / "cmds.txt").write_text("".join(l + "\n" for l in lines(d, [("a", timed(0.05)), ("b", "exit 3")])))
        base = [sys.executable, str(REPO / "scripts/run_queue.py"), str(d / "cmds.txt"), "--cwd", str(d), "--no-gpu-env", "--poll", "0.05",
                "--stagger", "0", "--retries", "0"]
        r = subprocess.run(base + ["--dry-run"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("would launch 2 now", r.stdout)
        self.assertFalse((d / "queue.log").exists()); self.assertFalse((d / "a").exists())
        r = subprocess.run(base + ["--sentinel", str(d / "q.done")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual((d / "a/.rc").read_text(), "0\n"); self.assertEqual((d / "b/.rc").read_text(), "3\n")
        self.assertIn("done b exit 3", (d / "queue.log").read_text()); self.assertTrue((d / "q.done.failed").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
