# 性能对比的口径更正：参考数字是怎么测出来的

背景：用户给出四种实现的对比表（第 4 列 `DeepInfer #92（DP8×EP8）`）。
我先前用自建 bench 打 OpenAI serve 层得到的数字比它低 3-5 倍，并据此推测"平台/后端问题"。
**这个推测是错的。** 查 `doc/sm90.md` 后确认：测法、并发口径、预填配置全都不同。

---

## 一、参考数字的真实来源（`doc/sm90.md` 原文）

> `examples/latency_benchmark.py`, DSpark on (`DEEPINFER_USE_NEXTN_DECODE=5`, `DEEPINFER_ENABLE_MTP_VARLEN=1`),
> `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`, `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`, CUDA graphs on.

### 最关键的一条：表里混了两个不同的并行配置

> | GPUs | Decode / throughput | Long-prompt prefill |
> | --- | --- | --- |
> | 8 | **DP8 x EP8** | **CP8 x EP8** |

- **TTFT 那几行（128K=2.114s、256K=4.310s、512K=9.032s、~1M=19.41s）是 CP8 × EP8**。
  CP8 把序列切到 8 张卡上算，预填天然快接近一个数量级。
- **Decode 那几行（158 / 2912 / 4259 tok/s、TPOT 6.33/6.13/8.32 ms）是 DP8 × EP8。**

### 预填调度：TTFT 与解码用的是不同策略

> Prefill scheduling trades latency for throughput: `DEEPINFER_PREFILL_ATTEMPT_INTERVAL` /
> `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL` (defaults **70 / 100** decode steps) batch prefills ...
> The single-request TTFT numbers below set **both to 1**.

所以：
- 单请求 TTFT 要 `=1`；
- 并发吞吐用默认 **70/100**（预填故意排在解码后面，这就是"吞吐优先"）。

### 解码测法

> random token ids, **1024 in / 512 out**, default prefill intervals;
> At 32 and 64 it is the total throughput over **4 x C requests with at most C in flight**, the median of 3 runs.
> **about 4.2-4.4 tokens are accepted per step**

最后一句说明参考数字是**开着 MTP 投机解码**的（每步接受 4.2-4.4 个 token）。

---

## 二、我的实测 vs 参考（并给出差异原因）

我测的是 OpenAI serve 层（`serve/server.sh`），`--dp 8 --cp`（无此参数）。

| 指标 | 参考（官方 harness） | 我的实测 | 差异的主因 |
|---|---|---|---|
| TTFT 128K | 2.114 s（**CP8**） | **10.16 s**（DP8，serve） | serve 层**没有 CP**，序列不切分，约 4.8× |
| TTFT 256K | 4.310 s（CP8） | 未测（KV 不够） | 同上 |
| Decode 单会话 | 158 tok/s（TPOT 6.33 ms） | **48.1 tok/s**（TPOT 20.8 ms） | 未开 `DEEPINFER_ENABLE_MTP_VARLEN=1` |
| Decode c=64 | 4259 tok/s（TPOT 8.32 ms） | **1062 tok/s**（TPOT 39.8 ms） | 同上 + 请求到达模式不同 |
| 预填边际吞吐 | ~60k tok/s（CP8 折算） | **13.3k tok/s**（DP8） | CP 差异 |

我的预填 scaling（DP8，serve，冷）：8K→1.16s、32K→2.41s、64K→4.96s、128K→10.16s，
**严格线性**，边际 13.3k tok/s。所以不是 O(n²)，也不是分块问题。

---

## 三、我犯的三个错误（记录以免重犯）

1. **把 `lookahead_tokens` 当成 MTP draft 深度。**
   `latency_benchmark.py:603` 自己也是 `min(1, DEEPINFER_USE_NEXTN_DECODE)` = 1。
   它是**调度**参数，不是投机深度。我改成 3 之后 c=1 从 48.1 掉到 39.2、c=64 从 1062 掉到 589 —— 纯粹是负收益。
   **MTP 真正的开关是 `DEEPINFER_ENABLE_MTP_VARLEN=1`**（`environ.py:137`，默认 False），我一次都没设过。

2. **误判 `triton_fp8` 后端"不支持 fp4"。**
   真实原因是 **OOM**：`Tried to allocate 13.56 GiB, 11.53 GiB free`。
   `doc/sm90.md:72` 写明 `triton_fp8` 是"FP8 upcast，约 2 倍专家显存，fits EP8 but not EP4"。
   我把 `--max-num-seqs 64 --max-model-len 131072` 配得太大，把专家显存挤掉了。

3. **把 c=32 的 18 秒 TTFT 当成调度 bug。**
   那正是 `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL` 默认 100 步的行为：预填故意被排到解码之后。
   官方并发测法只看总吞吐，不看 TTFT，所以这个数字在设计上就不是问题。

---

## 四、结论

**之前那组数字不构成对参考表的反驳，也不构成对 DeepInfer 的质疑。**
要真正复现，必须：

1. 用 `examples/latency_benchmark.py`（不是 OpenAI serve 层），因为它支持 `--cp`；
2. 设 `DEEPINFER_ENABLE_MTP_VARLEN=1`（+ 默认 `DEEPINFER_USE_NEXTN_DECODE=5`）；
3. 分两次跑：
   - TTFT：`--ep 8 --cp 8 --dp 1 --suites ttft`（CP8×EP8），并把两个 prefill interval 设为 1；
   - 解码：`--ep 8 --cp 1 --dp 8 --suites single,concurrency --concurrency 32 64`（DP8×EP8），用默认 prefill interval。
4. MoE 后端用默认 `auto`（humming，与文档一致；文档未指定其他后端）。

**尚未验证**：serve 层能否支持 CP。目前 `serve/server.sh` 没有 `--cp`，所以"对外服务形态"和"参考 TTFT 数字"用的是两种并行方式，
这一点需要单独决定（要么给 serve 层加 CP，要么接受 serve 层预填的 4.8× 差距）。
