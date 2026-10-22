# 别人怎么在 8×H20 上跑出那组数字的：SGLang 的 Engram 设计

起因：用户要求"先测冯骐那套 SGLang 优化"。查证结果与对 Engram 的关键发现。

## 一、那套 SGLang 不在我们能用的版本里

镜像里的 `sglang 0.5.22`：**没有 `DeepseekV41`、没有任何 engram 代码**。
DeepSeek-V4.1 的 SGLang 支持全在
[sglang PR #39114 "[DSV4.1] Enable Hopper decode optimizations for TP8"](https://github.com/sgl-project/sglang/pull/39114)：
**196 个文件、+22910/-1494 行、状态 open（未合并）**。

它的文件清单里含完整实现：

```
python/sglang/srt/layers/engram.py                       +972
python/sglang/kernels/ops/embeddings/engram_gather.py     +80
python/sglang/kernels/ops/embeddings/engram_hash.py      +363
python/sglang/srt/layers/attention/deepseek_v4_backend.py +2090
python/sglang/srt/entrypoints/openai/encoding_dsv41.py    +734
...
```

所以"测冯骐那套"= 构建这个未合并的 22k 行分支，是另一个工程，不是一次 benchmark。

## 二、但它的 Engram 算法比我们用的 PR#99 明显更省（这是重点）

**SGLang（PR#39114）：**

```python
values = self._owned_rows(indices)          # 只查自己分片，非本片行输出 0（一个 Triton kernel）
if self.tp_size > 1:
    values = tensor_model_parallel_all_reduce(values)   # 跨 rank 求和
```
`engram_gather.py` 注释原文：
> The table holds rows [row_lo, row_hi); an id outside it is not read and comes out as zeros,
> **which is what the sharded all-reduce sums**.

每个 rank 只查 `T × 24` 行（真实 token 数，**不 padding**），一次 all-reduce 拿到全部结果。

**我们的（DeepInfer PR#99 `engram_nvlink.py::fetch`）：**

```python
dist.all_gather_into_tensor(gathered, padded)   # padded 到 max_num_tokens，全部 indices 发给所有 rank
rows = table.index_select(0, where(owned, local_idx, 0).flatten())   # 查 world × T × 24 行
send.copy_(...); send.masked_fill_(~owned, 0)    # 其中 7/8 立即被清零
dist.all_to_all_single(recv, send)               # padding 后的整块行数据
```

每步多付三笔：
1. **padding 到 `max_num_decoding_tokens = max_num_seqs × (1+nextn)`**（解码时远大于真实 T）；
2. **8× 冗余查表**（查了 `world × T × 24` 行，7/8 立刻清零）；
3. `masked_fill_` + `copy_` + `all_to_all` 的多余显存往返。

## 三、这解释了多少

实测 per-step：我们 ~65 ms（accept≈5, TPOT 13-16 ms），参考 ~26 ms（accept≈4.3, TPOT 6.1-8.3 ms）。
上面三笔在解码步上的量级是每步几十 MB 到数百 MB 的无效流量 + 多个大 kernel，
**足以解释其中的相当一部分，但未做 profiler 归因，不能声称是全部**。

## 四、可移植的优化（纯 Python，不动引擎）

把 `fetch` 改成"**index all_to_all + 本地查表 + 行 all_to_all**"，保持现有 `(fp8_rows, sf)` 契约不变：

1. 不做 padding，按真实 `T` 分配；
2. 只把 indices 做 all_to_all（`T×24` 个 int32，微秒级），每个 rank 收到"别人要我的行"；
3. 只查这些行（1× 而非 8×），不需要 `masked_fill_` 清零；
4. 行数据 all_to_all 回传；对端按 owner 拼接。

这个方案比 SGLang 的 all-reduce 更省字节（fp8 行 vs bf16 值），且**保留 fp8 + sf 的契约**，
不需要改消费侧（`engram.py` / gate）。

替代方案（更简单但改契约）：照 SGLang 那样在本地反量化成 bf16、零填充、all_reduce，
但输出就从 `(fp8, sf)` 变成 bf16，会动到消费侧，风险更大。

## 五、建议顺序

1. 先做第四节的路由式优化，用 `tests/distributed/test_engram_nvlink.py` 验正确性（2 卡 5 秒），
   再用 `latency_benchmark.py --suites single,concurrency` 对比 44.3/1178.8/2156.6。
2. 若仍与参考有差距，再上 profiler 拆单步。
3. 想公平对比 SGLang 那套，需要单独构建 PR#39114 分支（工程量另计）。
