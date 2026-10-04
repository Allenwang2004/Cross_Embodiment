#!/usr/bin/env bash
# Start m2g and m2k continuation in parallel once the early beta-NN test and the
# long-search controls have released their CPU.
cd "$(dirname "$0")/.."
until grep -q "=== bnn m2c_lib done ===" outputs/train_logs/bnn_m2c_driver.log 2>/dev/null \
   && grep -q "=== long controls done ===" outputs/train_logs/long_control_m2c.log 2>/dev/null; do
  sleep 120
done
for p in m2g m2k; do
  EVALS=512 PAR=${PAR:-2} nohup bash scripts/run_continuation.sh $p outputs/continuation_clips.txt \
    > outputs/train_logs/continuation_${p}_driver.log 2>&1 &
done
wait
echo "=== m2g m2k done ==="
