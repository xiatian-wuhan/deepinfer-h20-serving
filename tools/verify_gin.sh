#!/bin/bash
# 20-second GIN availability probe. Does NOT touch glm53-flash and does NOT load weights.
# Runs DeepEP's own engram test on 2 GPUs (tiny tensors, ~600 MB/rank) so it fits in the
# ~1.3 GB/card left while production runs.
#
#   tools/verify_gin.sh            # default: GPUs 6,7
#   GPUS='"device=4,5"' tools/verify_gin.sh
#
# Verdict:
#   GIN OK    -> no "Unable to use GIN" warning, test prints its result
#   GIN DEAD  -> "Unable to use GIN: Peermem is not supported, and device N does not support DMA-BUF"
set -uo pipefail
ROOT=/home/ubuntu/data1/xia/deepinfer-h20
IMAGE=deepinfer-sm90:v41-sm90
GPUS=${GPUS:-'"device=6,7"'}
# PROCS=1 halves the CUDA-context footprint; useful when glm53-flash is warm and only
# ~0.5 GiB/card is free (2 procs then die with "Failed to CUDA calloc async").
PROCS=${PROCS:-2}
LOG=$ROOT/logs/e2e/gin_probe_$(date +%H%M%S).log
mkdir -p "$ROOT/logs/e2e"

echo "probing GIN on $GPUS procs=$PROCS  (log: $LOG)"
docker run --rm --gpus "$GPUS" --network host --device /dev/infiniband \
  --cap-add IPC_LOCK --ipc=host --shm-size=16g --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 -e NCCL_IB_HCA=mlx5_0 \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET -e PROCS \
  "$IMAGE" bash -lc '
    cd /opt/deepinfer/DeepInfer
    timeout 180 python vendor/DeepEP/tests/engram/test_engram.py \
      --num-processes '"$PROCS"' --num-entries-per-layer 1024,1024 --hidden 512 \
      --num-tokens 8 --num-entries-per-token 2
  ' > "$LOG" 2>&1
rc=$?

echo "--- verdict ---"
if grep -q 'Failed to CUDA calloc async\|ncclUnhandledCudaError' "$LOG"; then
    # Nothing to do with GIN: the probe died in init_process_group because the cards are full
    # (glm53-flash warm leaves ~0.5 GiB/card). Stop it or wait for a free window.
    echo "OOM  -- 显存不足，本次没有给出 GIN 结论（不是 GIN 失败）"
    grep -m1 'Failed to CUDA calloc async' "$LOG" | cut -c1-160
elif grep -q 'Unable to use GIN' "$LOG"; then
    echo "GIN DEAD  ($(grep -c 'Unable to use GIN' "$LOG") rank warning(s))"
    grep -m1 'Unable to use GIN' "$LOG" | cut -c1-160
elif grep -q 'NET/IB : No device found' "$LOG"; then
    echo "IB NOT VISIBLE IN CONTAINER (network/device flags wrong)"
elif grep -qE 'NET/IB.*Using \[0\]mlx5' "$LOG"; then
    if [ $rc -eq 0 ]; then echo "GIN OK  (exit 0, IB up, no GDR warning)"; else echo "IB UP, no GDR warning, but test exited $rc - see $LOG"; fi
else
    echo "INCONCLUSIVE (exit $rc) - see $LOG"
fi
echo "log: $LOG"
