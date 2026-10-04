#!/usr/bin/env bash
# alert when a live child-only run sets a new held-out low under 0.40, crashes,
# or when they all finish. Fields are pulled one at a time -- a single sed over
# the whole eval line mis-captured `test cost` because the line carries two
# "angle N" fields and the greedy match took the wrong one.
cd /home/allen19/crossenbodiment
declare -A best
best[child-lr3e-4]=0.3674
best[child-lr1e-3-pairs32]=0.3635
best[child-lr3e-4-pairs16]=0.3506
while true; do
  names=$(ps -eo args | grep "[t]rain_es.py" | grep -o "run-name [a-z0-9.-]*" | sed 's/run-name //' | sort -u)
  alert=""; cur=""
  for d in $(ls -d wandb/run-20260922_*-* 2>/dev/null); do
    n=$(python3 -c "import json;print(' '.join(json.load(open('$d/files/wandb-metadata.json')).get('args',[])))" 2>/dev/null | grep -o "child-lr[a-z0-9.-]*" | head -1)
    [ -z "$n" ] && continue
    echo "$names" | grep -qx "$n" || continue
    l=$(grep -E "\[eval @" "$d/files/output.log" 2>/dev/null | tail -1)
    [ -z "$l" ] && continue
    u=$(echo "$l"  | grep -oE "eval @ [0-9]+"   | grep -oE "[0-9]+")
    tr=$(echo "$l" | grep -oE "train cost=[0-9.]+" | cut -d= -f2)
    te=$(echo "$l" | grep -oE "test cost=[0-9.]+"  | cut -d= -f2)
    ag=$(echo "$l" | grep -oE "angle [0-9.]+" | head -1 | cut -d' ' -f2)
    row="$n u$u train $tr test $te ang ${ag}deg"
    cur="$cur$row"$'\n'
    b=${best[$n]:-9}
    if [ -n "$te" ] && awk -v a="$te" -v b="$b" 'BEGIN{exit !(a+0<0.40 && a+0<b+0)}'; then
      best[$n]=$te; alert="$alert NEWLOW $row"$'\n'
    fi
  done
  err=$(grep -lE "Traceback|CUDA out of memory" outputs/train_logs/child_only/lr*.log 2>/dev/null)
  [ -n "$err" ] && echo "ERROR in: $err"
  [ -n "$alert" ] && printf '%s' "$alert"
  [ -z "$names" ] && { printf 'ALL RUNS ENDED\n%s' "$cur"; break; }
  sleep 300
done
