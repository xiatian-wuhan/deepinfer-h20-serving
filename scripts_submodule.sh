#!/bin/bash
ROOT=/home/ubuntu/data1/xia/deepinfer-h20
LOG=$ROOT/logs/submodule.log
exec >> "$LOG" 2>&1
echo "=========== START $(date -Is) ==========="
cd "$ROOT/repos/deepinfer-sm90-92" || exit 1
for m in vendor/DeepSelect vendor/FlashMLA vendor/FlashMLA-SM90 vendor/DeepEP vendor/DeepGEMM; do
  echo "--- init $m"
  timeout 900 git submodule update --init --depth 1 "$m" && echo "[ok] $m $(git -C $m rev-parse --short=10 HEAD 2>/dev/null)" || echo "[FAIL] $m"
done
echo "=========== DONE $(date -Is) ==========="
