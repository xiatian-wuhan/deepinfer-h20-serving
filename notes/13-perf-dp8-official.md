# 性能实测（DP8×EP8，官方 harness）与参考表对比

窗口：2026-10-22 17:54 起（**看门狗已按用户要求关闭，恢复时机由用户决定**）
工具：`examples/latency_benchmark.py`（仓库自带权威基准，不是自建 bench）
环境：`EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1 DEEPINFER_ENABLE_MTP_VARLEN=1`
其它与 doc/sm90.md 一致：YOCO compact / KEEP_TOKENS=140 / CUDA graphs on / nextn 5 / chunk 32768
脚本自报 layout：`EP8 DP8 CP1 (8 GPUs), nextn 5, chunk 32768`

## 一、TTFT（DP8×EP8，两个 prefill interval 设 1）

| input tokens | 本机 DP8 实测 | 文档 CP8 | 文档 DP4 |
|---:|---:|---:|---:|
| 1024 | 0.345 | 0.186 | 0.185 |
| 4096 | 0.567 | 0.185 | 0.187 |
| 16384 | **1.466** | 0.302 | 0.499 |
| 65536 | **5.294** | 1.058 | 1.829 |
| 131072 | **10.641** | 2.114 | 3.689 |

**DP8 与 CP8 差约 5 倍，由并行方式决定，不是实现缺陷。**
用户表格里的 2.114 / 4.310 / 9.032 / 19.41 全部是 **CP8** 数字，把它们标成 "DP8×EP8" 是误记。

OpenAI serve 层测的 128K TTFT 是 10.16 s，官方 harness 10.64 s —— **服务层额外开销约 0**。

## 二、解码（DP8×EP8，默认 prefill intervals，MTP on）

| mode | 本机实测 | 参考表 | 比值 |
|---|---|---|---|
| single | 44.3 tok/s（TPOT 22.58 ms, accept 2.40） | 158（6.33 ms） | 3.6× |
| c=32 | 1178.8 tok/s（TPOT 12.97 ms, accept **5.04**） | 2912（6.13 ms） | 2.5× |
| c=64 | 2156.6 tok/s（TPOT 16.23 ms, accept **4.91**） | 4259（8.32 ms） | 2.0× |

## 三、已排除项与剩余嫌疑

已排除：
1. **不是 OpenAI 服务层**：同引擎 serve 48.1 tok/s vs harness 44.3；TTFT 10.16 vs 10.64。
2. **不是 MTP 未开**：`DEEPINFER_ENABLE_MTP_VARLEN=1` 是缺失开关；开启后 accept 在 c32/c64 达
   **5.04 / 4.91**，**高于参考的 4.2-4.4**。之前 0.00% 是漏设环境变量。
3. **不是 MoE 后端选错**：日志 `SM90 MoE (humming)`，与文档默认一致。
   （更正：我曾断言 triton_fp8 不支持 fp4，实为 **OOM**——文档写明它是 FP8 upcast、约 2 倍专家显存。）
4. **不是预填分块**：预填严格线性（DP8 边际 13.3k tok/s），非 O(n²)。
5. **不是 `lookahead_tokens`**：官方 harness 也用 `min(1, ...)`；改 3 是负收益（c=1 48.1→39.2、
   c=64 1062→589），确认它不是投机深度参数。

剩余首要嫌疑：**Engram 的 NVLink 取表路径（本机独有）**

参考数字来自 NCCL GIN（GPUDirect RDMA）可用的机器；本机不可用（notes/06），
走的是 DeepInfer PR#99 的 NVLink 回退。两者每步代价不同。

已定位一处明确的 8× 冗余（`engram_nvlink.py::fetch`）：

```python
dist.all_gather_into_tensor(self._gathered_indices, padded, group=self.group)  # 每个 rank 拿到所有 indices
local_idx = self._gathered_indices[:, layer_idx] - self.rank * num_entries      # (world, T, entries)
owned = (local_idx >= 0) & (local_idx < num_entries)
rows = table.index_select(0, torch.where(owned, local_idx, 0).flatten())        # 查 world 倍的行
send.copy_(rows.view(send.shape)); send.masked_fill_(~owned.unsqueeze(-1), 0)   # 7/8 立即被清零
```

每个 rank 查 `world × T × 24` 行，只有 1/8 是自己的。这是每步数百 MB 级无效显存流量。
但量级估计只值几 ms，**不足以解释 40 ms/step 的全部差距**，必须上 profiler 归因。

## 四、复现命令

```bash
# TTFT（DP8, prefill interval=1）
docker exec di-e2e bash -lc 'cd /opt/deepinfer/DeepInfer &&
  EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1 DEEPINFER_ENABLE_MTP_VARLEN=1 \
  DEEPINFER_PREFILL_ATTEMPT_INTERVAL=1 DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL=1 \
  python -u examples/latency_benchmark.py --ep 8 --cp 1 --dp 8 --suites ttft \
    --ttft_lengths 1024 4096 16384 65536 131072 --prompt random --out_dir /e2e-logs/b-ttft-dp8'

# 解码（DP8, 默认 prefill interval）
docker exec di-e2e bash -lc 'cd /opt/deepinfer/DeepInfer &&
  EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1 DEEPINFER_ENABLE_MTP_VARLEN=1 \
  python -u examples/latency_benchmark.py --ep 8 --cp 1 --dp 8 \
    --suites single,concurrency --concurrency 32 64 --prompt random --out_dir /e2e-logs/b-dec-dp8'
```

## 五、下一步（按价值）

1. **profile 单步**：把 65 ms 拆成 engram fetch / MoE / attention / 其它。
2. **优化 PR#99 的 fetch**：改为「all_to_all 分发 indices → 只查自己分片 → all_to_all 回传」，
   去掉 8× 冗余。纯 Python，不动引擎。
3. 若要 TTFT 对齐 CP8：给 serve 层补 `--cp`（`_config.py` 已支持 `cp_size`，serve 未透传）。
