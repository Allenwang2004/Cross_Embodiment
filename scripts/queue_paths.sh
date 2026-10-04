#!/usr/bin/env bash
# Run continuation on the three new paths once m2c's has finished, so the CPU
# goes to the controls and m2c first (they decide whether the result is real).
cd "$(dirname "$0")/.."
until grep -q "=== continuation m2c done ===" outputs/train_logs/continuation_m2c_driver.log 2>/dev/null; do sleep 60; done
for p in m2s m2g m2k; do
  EVALS=512 PAR=${PAR:-3} bash scripts/run_continuation.sh $p outputs/continuation_clips.txt \
    > outputs/train_logs/continuation_${p}_driver.log 2>&1
done
echo "=== queued paths done ==="
