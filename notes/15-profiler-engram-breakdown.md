# Profiler 归因：单步 59 ms 里 71% 是 Engram 的 NVLink 取表

工具：torch.profiler（kineto，CUDA+CPU），只挂 rank0
方法：`tools/prof_bench.py` — import 前设好 argv，monkey-patch `DeepInferDriver.step`，
      profile 第 10..39 步（共 30 个解码步），跑完直接打印完整 kernel 名与占比（不依赖 trace 文件）。

配置：`EP8 DP8 CP1`，`DEEPINFER_ENABLE_MTP_VARLEN=1`，CUDA graphs on，humming MoE。

## 一、结果

30 步，每步平均 **58.9 ms** GPU 时间（`deepinfer:launch_forward_N1_B1` 1766.3 ms / 30）。

| kernel（完整名） | 总耗时 | 次数/步 | 每步 | 归属 |
|---|---|---|---|---|
| `masked_fill_kernel` | 398.89 ms (11.4%) | 4 | **13.30 ms** | Engram：把 7/8 非本片行清零 |
| `direct_copy_kernel_cuda` | 334.32 ms (9.6%) | 4 | **11.14 ms** | Engram：拷进 padding 的 send 缓冲 |
| `vectorized_gather_kernel<16,int>` | 261.66 ms (7.5%) | 2 | **8.72 ms** | Engram：8× 冗余查表 |
| `ncclDevKernel_SendRecv` | 261.65 ms (7.5%) | 1 | **8.72 ms** | Engram：all_to_all |
| `_block32_gemm_kernel` | 94.55 ms (2.7%) | 262 | 3.15 ms | MoE / FP8 GEMM |
| `flash_fwd_splitkv_mla_fp8_sparse` | 23.69 ms | 38 | 0.79 ms | MLA 稀疏注意力 |
| `sm90_tf32_hc_prenorm_gemm` | 22.32 ms | 86 | 0.74 ms | mHC prenorm |
| `nvjet_sm90_tss_*` | 18.16 ms | 2 | 0.61 ms | cuBLAS/cuDNN |
| `deep_ep::ep::dispatch_impl` | 17.40 ms | 40 | 0.58 ms | EP 分发 |
| `deep_ep::ep::combine_impl` | 12.23 ms | 40 | 0.41 ms | EP 归并 |
| `_norm_kernel` / `mhc_*` | 42.4 ms | 132+2580 | 1.41 ms | 归一化 / mHC |
| `flash_fwd_mla_combine` | 9.46 ms | 43 | 0.32 ms | 注意力收尾 |
| `launch_sample` / `launch_dflash_decode_forward` | 114.8 ms | 2+3 | 3.83 ms | 采样 / DFlash |

## 二、Engram 取表 = 41.9 ms/步 = **71%**

```
13.30 (masked_fill 清零) + 11.14 (direct_copy) + 8.72 (8× 查表) + 8.72 (all_to_all) = 41.88 ms
41.88 / 58.9 = 71.1%
```

**其中 33.1 ms（前三项）是纯算法冗余**，真正的跨卡传输只需要 8.7 ms。
注意力只有 0.8 ms/步，MoE GEMM 3.2 ms —— 都不是瓶颈。

## 三、和参考对上了

| | per-step |
|---|---|
| 参考（文档，DP8×EP8，GIN 路径） | **26 ms**（accept 4.3 × TPOT 6.33 ms） |
| 本机实测 | 58.9 ms |
| 本机减去 Engram 冗余（33.1 ms） | **25.8 ms** |

25.8 ≈ 26。这不是巧合：把 PR#99 回退实现的 padding + 8× 冗余 + masked_fill/copy 去掉，
per-step 就应该落回参考的量级。

## 四、为什么会有这些冗余（代码级）

`deepinfer/model_executor/parallel_utils/engram_nvlink.py::fetch`：

```python
padded = indices.new_full((num_layers, self.max_num_tokens, ...), -1)   # ① padding 到 max_num_decoding_tokens
dist.all_gather_into_tensor(self._gathered_indices, padded)             # ② 全部 indices 发给所有 rank
local_idx = self._gathered_indices[:, layer_idx] - self.rank * num_entries
owned = (local_idx >= 0) & (local_idx < num_entries)
rows = table.index_select(0, torch.where(owned, local_idx, 0).flatten())  # ③ 查 world 倍的行（vectorized_gather）
send.copy_(rows.view(send.shape))                                        # ④ direct_copy
send.masked_fill_(~owned.unsqueeze(-1), 0)                              # ⑤ masked_fill 清零 7/8
dist.all_to_all_single(self._recv, self._send)                           # ⑥ 真正的传输
```

①~⑤ 就是那 33.1 ms。`max_num_decoding_tokens = max_num_seqs × (1 + DEEPINFER_USE_NEXTN_DECODE)`，
解码时远大于真实 T，padding 放大了 ③④⑤ 的规模。

## 五、修复方向（有数据支撑，不再是猜）

照 SGLang PR#39114 的思路，把 fetch 改成：

1. **不 padding**，按真实 `T` 分配；
2. indices 做 all_to_all（`T×24` 个 int32，微秒级），每个 rank 只收到"别人要我的行"；
3. **只查自己要发的行**（1× 而非 8×），**不需要 masked_fill 清零、不需要整块 direct_copy**；
4. 行数据 all_to_all 回传，按 owner 拼接，保持 `(fp8_rows, sf)` 契约不变。

预期：去掉 ①③④⑤ 的 ~33 ms/步，per-step 回到 ~26 ms，单会话解码从 44.3 tok/s 回到 150+ tok/s。
正确性用 `tests/distributed/test_engram_nvlink.py`（2 卡 5 秒）验证，再用
`latency_benchmark.py --suites single,concurrency` 复测。

## 六、复现

```bash
docker cp tools/prof_bench.py di-e2e:/tmp/prof_bench.py
docker exec -d di-e2e bash -lc 'cd /opt/deepinfer/DeepInfer &&
  EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1 DEEPINFER_ENABLE_MTP_VARLEN=1 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 python -u /tmp/prof_bench.py > /e2e-logs/prof2.log 2>&1'
grep "^\[prof\]" logs/e2e/prof2.log
```
