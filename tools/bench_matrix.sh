#!/bin/bash
# README §五 那张表的全部测量。在容器内跑（要 CUDA + 引擎），一次约 25 分钟。
#
#   docker cp tools/bench_matrix.sh di-e2e:/tmp/bench_matrix.sh
#   docker exec -d di-e2e bash -lc 'bash /tmp/bench_matrix.sh /e2e-logs/r3 > /e2e-logs/r3.log 2>&1'
#
# 每个 --out_dir 由 harness 写 summary.md + summary.json，data/ 里冻结的就是这两份。
# 环境变量取自启动服务时的那一份（/e2e-logs/serve.env），保证和生产同配置：
# MTP 深度、VARLEN、YOCO 这些会直接改变数字。
set -uo pipefail

OUT=${1:-/e2e-logs/bench}
cd /opt/deepinfer/DeepInfer || exit 1
PY=/opt/deepinfer/venv/bin/python
mkdir -p "$OUT"

if [[ -r /e2e-logs/serve.env ]]; then
    grep -E '^(DEEPINFER_|NCCL_IB_HCA|CUDA_VISIBLE_DEVICES)' /e2e-logs/serve.env > /tmp/bench.env
    set -a; source /tmp/bench.env; set +a
fi
echo "### nextn=${DEEPINFER_USE_NEXTN_DECODE:-unset} varlen=${DEEPINFER_ENABLE_MTP_VARLEN:-unset} gin=${EP_DISABLE_GIN:+disabled}"

bench() { echo; echo "### $(date -u +%FT%TZ) $*"; $PY -u examples/latency_benchmark.py --prompt random "$@"; echo "### rc=$?"; }
decode_args=(--suites single,concurrency)  # 解码用 harness 默认的 4096 in / 1024 out

# 生产布局 CP8（注意力 TP8 + 专家 EP8，8 卡同一个 batch）
bench --ep 8 --cp 8 --suites ttft --ttft_lengths 131072 --ttft_warmup 1 --ttft_repeats 3 --out_dir "$OUT/ttft-cp8-128k"
# 1M 预填单次要 20 s，预热一次太贵：单次测量
bench --ep 8 --cp 8 --suites ttft --ttft_lengths 1040384 --ttft_warmup 0 --ttft_repeats 1 --out_dir "$OUT/ttft-cp8-1m"
bench --ep 8 --cp 8 "${decode_args[@]}" --concurrency 32 64 --out_dir "$OUT/dec-cp8"

# DP8（8 个独立副本，一个请求只吃一张卡的注意力），用来对比布局
bench --ep 8 --cp 1 --dp 8 --suites ttft --ttft_lengths 131072 --ttft_warmup 1 --ttft_repeats 3 --out_dir "$OUT/ttft-dp8-128k"
bench --ep 8 --cp 1 --dp 8 --suites ttft --ttft_lengths 1040384 --ttft_warmup 0 --ttft_repeats 1 --out_dir "$OUT/ttft-dp8-1m"
bench --ep 8 --cp 1 --dp 8 "${decode_args[@]}" --concurrency 32 64 --out_dir "$OUT/dec-dp8"

# A/B：关掉 GIN，走 PR#99 的 NVLink Engram 回退路径（旧笔记里的数字是这个配置）
if [[ "${SKIP_NVLINK:-0}" != "1" ]]; then
    export EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1
    echo
    echo "### nvlink arm: EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1"
    bench --ep 8 --cp 8 --suites ttft --ttft_lengths 131072 --ttft_warmup 1 --ttft_repeats 3 --out_dir "$OUT/ttft-cp8-128k-nvlink"
    bench --ep 8 --cp 8 "${decode_args[@]}" --concurrency 32 --out_dir "$OUT/dec-cp8-nvlink"
fi

echo
echo "### done $(date -u +%FT%TZ)"
