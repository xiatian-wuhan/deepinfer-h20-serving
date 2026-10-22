# 17 — Engram NVLink fetch：把 padding 从 32768 降到真实 token 数

日期：2026-10-22。文件 `deepinfer/model_executor/parallel_utils/engram_nvlink.py`（PR#99 引入）。
背景见 notes/15（profiler 归因）与 notes/16（布局 A/B）。

## 实测事实（探针打出来的，不是推断）

给 `fetch` 加一行打印后跑 DP8 解码，8 个 rank 的前几次调用：

```
rank=* call=1 num_tokens=32768 max=32768 E=24 layers=2 send_MB=3072 table_MB=23438   # 预填
rank=* call=2 num_tokens=24    max=32768 E=24 layers=2 send_MB=3072 table_MB=23438   # 解码第 1 步
rank=* call=3 num_tokens=24    max=32768 ...
```

推论：

1. `max_num_tokens = 32768`（= `max_num_batched_tokens`），而解码一步只有 **24** 个 token
   （4 请求 × (1+5)）。**padding 放大 1365 倍**。
2. `_send` / `_recv` 各 **3072 MB → 合计 6 GB 显存**，纯粹因为按 32768 预留。
3. 各 rank 的 `num_tokens` 完全一致（都 24），且 `fetch` 每步都在 Python 侧执行
   （打印每步都出现）——不在 CUDA graph 内，所以可以做一次小的主机侧形状协商。
4. `E = num_entries_per_token = 24`，`hidden = 256`，fp8 表分片后 23.4 GB/rank。

## 三个缺陷

原实现每层做的事：

```python
padded = indices.new_full((layers, max_num_tokens, E), -1)   # 32768 行
all_gather_into_tensor(self._gathered_indices, padded)        # world × 32768 行
rows = table.index_select(0, where(owned, local_idx, 0).flatten())  # 再从 pinned host 抓 world×32768×E 行
send.copy_(rows.view(send.shape))                            # 3 GB 拷一遍
send.masked_fill_(~owned.unsqueeze(-1), 0)                   # 3 GB 再写一遍
all_to_all_single(self._recv, self._send)                    # 3 GB 再搬一遍
```

- **缺陷 1（放大）**：token 维按容量而不是真实长度预留。
- **缺陷 2（多余拷贝）**：`rows` 先分配再 `copy_` 进 `send`，两次 3 GB 往返。
- **缺陷 3（完全无用的 masked_fill）**：接收侧只读自己需要的那一片：
  `fetched = _recv[owner, layer, token_idx, entry_idx]`，`owner` 精确指出哪个 rank 拥有该行。
  非本 rank 拥有的槽位**永远不会被读到**，清零纯属浪费 13.3 ms。

## 改法

1. 每步先协商一次节点内最大 token 数（1 元 all_reduce MAX + 一次 `.item()`），
   按它分配/复用三个缓冲。DP 各副本批次不同也能对齐；相同则零额外开销。
2. 缓冲改成 **layer-major** `(layers, world, cap, E, row_bytes)`，让 `send[layer]` 连续，
   可以用 `torch.index_select(..., out=send.view(-1, row_bytes))` **直接写进发送缓冲**，
   消掉中间张量和一次 3 GB 拷贝。
3. 删掉 `masked_fill_`（见缺陷 3 的论证）；`all_to_all_single` 改为按层调用
   （维度 0 仍是 world，切分语义不变）。

缓冲尺寸变为 `layers × world × cap × E × 256B`：cap=24 时是 **2.4 MB**（原来 3 GB）。

## 正确性

新增 `tests/distributed/test_engram_nvlink.py`，用「全表 all_gather + 朴素索引」做参考实现，
覆盖本地/远程归属、两层不同表大小、**各 rank token 数不同**、以及 sf 复制表：

```
torchrun --nproc_per_node=2 tests/distributed/test_engram_nvlink.py
[test] PASS (0 mismatching layer-cases)
```

这一步是必须的：去掉 `masked_fill` 的正确性论证不是"看起来没事"，而是靠独立参考实现逐一比对字节。

## 实测效果（同机同 commit 63b430a，同 env）

| 指标 | 修复前 | 修复后 | 参考（`doc/sm90.md`） |
|---|---:|---:|---:|
| DP8 c=1 解码 | 86.3 tok/s | **151.9** | 158 |
| DP8 c=1 TPOT | 11.5 ms | **6.58 ms** | 6.33 ms |
| DP8 c=32 总吞吐 | 1615.7 | **2380.6** | 2912 |
| DP8 c=32 单请求 | 67.2 | **103.4** | — |
| DP8 c=32 TPOT | 14.98 ms | **9.70 ms** | — |
| DP8 c=64 总吞吐 | 2770.6 | **3826.7** | 4259 |
| CP8 c=1 解码 | ~212 | **240.6-279.1** | — |
| CP8 c=32 总吞吐 | 1720.9 | **1777.1** | — |

DP8 c=1 达到参考的 96%，TPOT 差 4%。收益不对称（DP8 +76%，CP8 +15%）是预期的：
CP8 时 8 张卡同跑一个 batch，每卡 token 少、padding 浪费本来就小。

## 一个踩到的坑：不要按「阶段」切 cap

第一版把 cap 写成 `max_num_decoding_tokens if num_tokens <= it else max_num_tokens`，
在 CP8 下没问题，但 **DP8 直接死锁**：8 卡 100% 空转、日志停住。
原因是 DP 各副本独立，一个副本可能在预填（cap=32768）而另一个在解码（cap=24），
`all_gather` 形状不一致，NCCL 永久等待。

最终的 `cap` 必须是**不依赖当步负载**的量。现在用的是 `max_num_tokens`（阶段无关、恒定）。
进一步压小的唯一安全前提是「组内所有 rank 跑同一个 batch」，即 `cp>1, dp=1`；
那需要在 `allocate_engram` 里把 cp 大小传进来才能证明，尚未做（见 notes/18 的后续项）。

附带修掉的一个运维坑：`pkill -f "multiprocessing.spawn"` 会把自己的命令行也匹配上而自杀，
旧 worker 存活继续占 88 GB 显存，必须写成 `pkill -9 -f 'multiprocessing[.]spawn'`。
