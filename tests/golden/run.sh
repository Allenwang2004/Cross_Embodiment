#!/bin/bash
# tests/golden/run.sh LABEL [name ...] -- run the golden cases (tests/golden/cases.txt; all of them, or the named ones) into
# outputs/refactor_golden/LABEL/<name>. Shared machine (CLAUDE.md): before every start it checks `free -g` (available) and the CPU
# idle share (top), and starts a case only if
#   - fewer than GOLDEN_P cases of this run are running (default 2),
#   - available - the case's memory estimate >= GOLDEN_MIN_AVAIL GB (default 50; estimates: g1_* 30 GB, others 12 GB),
#   - CPU idle >= 15 %, and, while idle < 25 %, none of this run's cases is running (sequential when the box is busy).
# If available drops below 20 GB while cases run, the most recently started case is killed (its process group), no new case
# starts, and run.sh exits 2. Every case runs in its own session (setsid) with the GPU-2 environment below; its stdout/stderr go
# to <case>/stdout.log, its real exit code to <case>/.rc and to run.log (rc=$? right after the command).
# A case whose folder already holds .rc = 0 is skipped (GOLDEN_FORCE=1 re-runs it); any other existing folder of that case is
# moved to outputs/refactor_golden/LABEL/_stale/ first. Exit code: 0 if every requested case exited 0, 1 otherwise, 2 on the
# memory stop.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[ "$REPO" != "/home/allen19/crossembodiment" ] || { echo "refusing to run in the main tree" >&2; exit 3; }
[ $# -ge 1 ] || { echo "usage: $0 LABEL [name ...]" >&2; exit 3; }
LABEL=$1; shift
cd "$REPO" || exit 3
export CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export UV_NO_SYNC=1            # `uv run` must not touch the shared .venv (a symlink into the main tree)
P=${GOLDEN_P:-2}; MIN_AVAIL=${GOLDEN_MIN_AVAIL:-50}; FORCE=${GOLDEN_FORCE:-0}
OUT=outputs/refactor_golden/$LABEL; LOG=$OUT/run.log; mkdir -p "$OUT"

ts() { date "+%m-%d %H:%M:%S"; }
log() { echo "$(ts) $*" | tee -a "$LOG"; }
avail() { free -g | awk '/^Mem:/{print $7}'; }
idle() { top -bn2 -d 2 | grep "%Cpu(s)" | tail -1 | awk -F, '{for (i = 1; i <= NF; i++) if ($i ~ / id/) {split($i, a, " "); print int(a[1])}}'; }
est() { case $1 in g1_*) echo 30;; *) echo 12;; esac; }

# the cases to run, in cases.txt order
declare -a NAMES CMDS
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    name=${line%%|*}; cmd=${line#*|}
    if [ $# -gt 0 ]; then
        keep=0; for want in "$@"; do [ "$want" == "$name" ] && keep=1; done; [ $keep == 1 ] || continue
    fi
    NAMES+=("$name"); CMDS+=("${cmd//\$LABEL/$LABEL}")
done < tests/golden/cases.txt
if [ $# -gt 0 ] && [ ${#NAMES[@]} -ne $# ]; then echo "unknown case among: $*" >&2; exit 3; fi
log "run.sh $LABEL: ${#NAMES[@]} cases, at most $P at once, keep >= $MIN_AVAIL GB available | git $(git rev-parse --short HEAD) $(git status --porcelain -- model scripts utils | wc -l) dirty files in model/ scripts/ utils/"

declare -A PID=() START=()  # running cases: subshell pid, start time (s)
ORDER=()                   # start order of the running cases (the last one is killed first)
fail=0; stopped=0

reap() {
    local n rc
    for n in "${!PID[@]}"; do
        if ! kill -0 "${PID[$n]}" 2>/dev/null; then
            wait "${PID[$n]}" 2>/dev/null
            rc=$(cat "$OUT/$n/.rc" 2>/dev/null || echo missing)
            [ "$rc" == "0" ] || fail=1
            unset "PID[$n]"
            local keep=() o; for o in "${ORDER[@]}"; do [ "$o" == "$n" ] || keep+=("$o"); done; ORDER=("${keep[@]}")
        fi
    done
}

guard() {                  # emergency stop: available < 20 GB -> kill the newest case of this run
    local a n i
    a=$(avail)
    if [ "$a" -lt 20 ] && [ ${#PID[@]} -gt 0 ]; then
        for ((i = ${#ORDER[@]} - 1; i >= 0; i--)); do n=${ORDER[$i]}; [ -n "${PID[$n]:-}" ] && break; done
        log "MEMORY STOP: available $a GB < 20 GB; killing $n (newest case), no new case will start"
        kill -TERM -- "-$(cat "$OUT/$n/.pgid")" 2>/dev/null; sleep 5; kill -KILL -- "-$(cat "$OUT/$n/.pgid")" 2>/dev/null
        stopped=1
    fi
}

for idx in "${!NAMES[@]}"; do
    name=${NAMES[$idx]}; cmd=${CMDS[$idx]}; d=$OUT/$name
    [ $stopped == 1 ] && break
    if [ -d "$d" ]; then
        if [ "$FORCE" != 1 ] && [ "$(cat "$d/.rc" 2>/dev/null)" == "0" ]; then log "skip $name (already done, .rc 0)"; continue; fi
        mkdir -p "$OUT/_stale"; mv "$d" "$OUT/_stale/$name.$(date +%s)"
    fi
    need=$(est "$name"); waited=0
    while :; do
        reap; guard; [ $stopped == 1 ] && break
        a=$(avail); i=$(idle); k=${#PID[@]}
        if [ "$k" -lt "$P" ] && [ $((a - need)) -ge "$MIN_AVAIL" ] && [ "$i" -ge 15 ] && { [ "$i" -ge 25 ] || [ "$k" -eq 0 ]; }; then break; fi
        [ $((waited % 300)) == 0 ] && log "waiting to start $name: running $k, available $a GB (needs $need + $MIN_AVAIL), CPU idle $i %"
        sleep 10; waited=$((waited + 10))
    done
    [ $stopped == 1 ] && break
    mkdir -p "$d"
    log "start $name (available $a GB, CPU idle $i %, running $k)"
    (
        t0=$(date +%s)
        setsid -w bash -c 'echo $$ > "$1/.pgid"; eval "$2"' _ "$d" "$cmd" > "$d/stdout.log" 2>&1
        rc=$?
        echo "$rc" > "$d/.rc"
        echo "$(ts) done $name exit $rc $(( $(date +%s) - t0 )) s" >> "$LOG"
    ) &
    PID[$name]=$!; START[$name]=$(date +%s); ORDER+=("$name")
    sleep 5
done
while [ ${#PID[@]} -gt 0 ]; do reap; guard; sleep 5; done

summary=""
for name in "${NAMES[@]}"; do
    rc=$(cat "$OUT/$name/.rc" 2>/dev/null || echo "not run"); summary+=" $name=$rc"
    [ "$rc" == "0" ] || fail=1
done
log "finished $LABEL:$summary"
[ $stopped == 1 ] && exit 2
exit $fail
