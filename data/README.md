# 测量数据

`README.md §五` 的每个数字都来自这里：`examples/latency_benchmark.py` 自己写出的
`summary.json` / `summary.md`，原样冻结，没有二次加工。

## 测量条件

| 项 | 值 |
|---|---|
| 主机 | 8 × H20-3e（141 GB HBM3e/卡），`k8s-node-238` |
| 驱动 | 595.91.07 **开源内核模块**（GIN 可用，见 `notes/23`） |
| 镜像 | `deepinfer-sm90:v41-sm90` |
| 代码 | DeepInfer `e205d42` + `patches/deepinfer-h20-full.patch` |
| 环境 | `DEEPINFER_USE_NEXTN_DECODE=5`、`DEEPINFER_ENABLE_MTP_VARLEN=1`、`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`、`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` |
| 时间 | 2026-10-23 |

- TTFT：128K 是 1 次预热 + 3 次测量取中位数；1M 预填一次 ~30 s，只测一次（`--ttft_warmup 0 --ttft_repeats 1`）。
- 解码：harness 默认的 4096 in / 1024 out，`ignore_eos` 生成满长度，TPOT/吞吐才有意义。
  不要改小输出长度——固定开销摊薄变少，聚合吞吐会假性下降。
- `--prompt random`（与 `coding` 的差异只在 token 分布，长度一致）。

## 目录

| 目录 | 内容 |
|---|---|
| `ttft-cp8-128k/` | 生产布局 CP8，131072 token 输入 |
| `ttft-cp8-1m/` | CP8，1040384 token 输入（单次） |
| `dec-cp8/` | CP8，c=1 / 32 / 64 |
| `ttft-dp8-128k/`、`ttft-dp8-1m/`、`dec-dp8/` | 同一组测量换成 DP8（8 个独立副本） |
| `ttft-cp8-128k-nvlink/`、`ttft-cp8-1m-nvlink/`、`dec-cp8-nvlink/` | A/B：`EP_DISABLE_GIN=1`，走 PR#99 的 NVLink Engram 回退路径 |
| `single-cp8-gin/`、`single-cp8-nvlink/` | 单会话解码的稳定性检查：同一配置连测 5 次 |
| `protocol/` | `serve_test.py`（26 条断言）、`cache_hit_matrix.py`（三协议缓存字段）、`recipe_parity.py`（与官方实现逐 token 对表）、`docker/verify_service.py`（镜像自检）的输出 |

每个目录里 `summary.md` 是 harness 自己写的元信息 + 表格（布局、commit、GPU、每条请求），
`summary.json` 是同样内容的机器可读版（每条请求的明细在 harness 的 `ttft.json` / `c32.json` 里，
按上面的命令重跑即可再生成，所以没进仓库）。

## 生成表格

```bash
python3 tools/summarize_data.py data/r2
```

## 复跑

```bash
docker cp tools/bench_matrix.sh di-e2e:/tmp/bench_matrix.sh
docker exec -d di-e2e bash -lc 'bash /tmp/bench_matrix.sh /e2e-logs/r3 > /e2e-logs/r3.log 2>&1'
```

脚本会从 `/e2e-logs/serve.env` 读回生产的那份环境变量（MTP 深度、VARLEN 这些直接改数字），
按 CP8 → DP8 → 关 GIN 三条臂依次跑，约 25 分钟。
