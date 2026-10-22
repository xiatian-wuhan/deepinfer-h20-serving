#!/bin/bash
# GPU test window manager with a hard watchdog.
#
#   tools/e2e_window.sh start <minutes> [extra docker -e ...]
#   tools/e2e_window.sh stop
#   tools/e2e_window.sh status
#
# WHY: the production service (glm53-flash) must be back up at an agreed time. Restoring it
# takes ~4.5 min (init engine 140 s + graph capture 74 s measured on 2026-10-22), so the
# watchdog restores it *minutes before* the deadline rather than at the deadline.
# The watchdog is a detached process, so it fires even if this shell dies.
set -uo pipefail
ROOT=/home/ubuntu/data1/xia/deepinfer-h20
MODEL=/home/ubuntu/data1/models/DeepSeek-V4.1-Flash-1017
IMAGE=deepinfer-sm90:v41-sm90
NAME=di-e2e
WD=$ROOT/logs/e2e/watchdog.log

restore() {
    echo "$(date '+%F %T') restore: killing $NAME" >>"$WD"
    docker rm -f "$NAME" >/dev/null 2>&1
    for _ in $(seq 1 30); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [ "$used" -lt 2000 ] && break
        sleep 2
    done
    echo "$(date '+%F %T') restore: docker start glm53-flash" >>"$WD"
    docker start glm53-flash >>"$WD" 2>&1
    for _ in $(seq 1 60); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18881/v1/models)
        [ "$code" = "200" ] && { echo "$(date '+%F %T') restore: glm53-flash READY (HTTP 200)" >>"$WD"; return 0; }
        sleep 10
    done
    echo "$(date '+%F %T') restore: FAILED to become ready" >>"$WD"
}

case "${1:-}" in
  start)
    MIN=${2:?usage: start <minutes> [extra docker args]}
    shift 2
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
        echo "error: $NAME already running" >&2; exit 1
    fi
    # watchdog: fire at MIN-6 minutes so the ~4.5 min restore still lands before the deadline
    LEAD=$(( MIN > 8 ? 6 : MIN / 2 ))
    ( sleep $(( (MIN - LEAD) * 60 )); restore ) >/dev/null 2>&1 &
    WD_PID=$!
    echo "$(date '+%F %T') start: window=${MIN}min lead=${LEAD}min watchdog_pid=$WD_PID" >>"$WD"

    echo "stopping glm53-flash ..."
    docker stop -t 40 glm53-flash >/dev/null
    docker rm -f "$NAME" >/dev/null 2>&1
    # --network host + /dev/infiniband are REQUIRED: on the default bridge network NCCL
    # reports "NET/IB : No device found" and falls back to Socket, so DeepEP Engram
    # (which needs NCCL GIN/GDAKI) cannot initialize. See notes/06-engram-gin-rootcause.md
    docker run -d --name "$NAME" --gpus all --network host --device /dev/infiniband \
      --cap-add IPC_LOCK --ipc=host --shm-size=64g \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v "$MODEL":/model:ro -v "$ROOT/logs/e2e":/e2e-logs \
      -e DEEPINFER_ACCELERATOR=nvidia -e DEEPINFER_CUDA_ARCH=sm90 \
      -e TORCH_CUDA_ARCH_LIST=9.0a \
      -e DEEPINFER_HF_MODEL_DIR=/model \
      -e DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact \
      -e DEEPINFER_PREFILL_STORE_EXTRA_STATE=1 \
      -e DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140 \
      -e NCCL_NVLS_ENABLE=0 -e NCCL_IB_HCA=mlx5_0 \
      -e PYTHONUNBUFFERED=1 \
      "$@" \
      "$IMAGE" sleep infinity >/dev/null
    echo "container $NAME up; logs bind-mounted at $ROOT/logs/e2e"
    echo "watchdog restores glm53-flash in $((MIN - LEAD)) min (pid $WD_PID; log $WD)"
    ;;
  stop)
    pkill -f "e2e_window.sh start" 2>/dev/null
    restore
    echo "restored; see $WD"
    ;;
  status)
    date '+%F %T'
    docker ps --filter "name=$NAME"  --format '  test:  {{.Names}} {{.Status}}'
    docker ps --filter "name=glm53-flash" --format '  prod:  {{.Names}} {{.Status}}'
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
    echo
    tail -5 "$WD" 2>/dev/null | sed 's/^/  wd: /'
    ;;
  *) echo "usage: $0 {start <minutes> [docker args]|stop|status}" >&2; exit 2 ;;
esac
