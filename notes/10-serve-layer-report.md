# OpenAI 服务层测试报告（功能 + 缓存 + 性能）

窗口：2026-10-22 16:50 - 17:05（**看门狗已由用户指示手动取消**，恢复时间由 agent 控制）
模型：DeepSeek-V4.1-Flash-1017，8×H20-3e，EP8/DP8，`EP_DISABLE_GIN=1` + PR#99 NVLink Engram

启动命令：

```bash
serve/server.sh start --dp 8 --port 8888 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --max-model-len 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 --hicache-size-gb 8
```

冷启动到 `/health` 200：**90 s**（第二次；首次 4 min，含 TileLang JIT）。

---

## 一、功能验证：14 项全通过

`tools/serve_test.py`

| 组 | 结果 |
|---|---|
| `/v1/models` | ✅ 返回 `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| 非流式对话 | ✅ 内容正确（"KV cache 是把 Transformer ... 已计算的 Key 和 Value 张量缓存下来..."）、`finish_reason=stop` |
| usage 字段 | ✅ `prompt_tokens / completion_tokens / total_tokens` |
| **`prompt_tokens_details.cached_tokens`** | ✅ vLLM 兼容字段，非流式与流式都有 |
| **流式 SSE** | ✅ `[DONE]`、首帧 `role=assistant`、逐 token delta（12 chunks）、`stream_options.include_usage` 生效 |
| 流式输出正确性 | ✅ 问"数一下 1 到 5"，返回 `'1 2 3 4 5'` |
| **缓存命中（API 层）** | ✅ 同一 4019-token 请求第二次 `cached_tokens=3840`（95.5%） |
| `/deepinfer/metrics` | ✅ 24 条 hicache 指标（按 dp_rank）+ 请求计数 |

### 我引入过的一个 bug（已修，`a2377a6`）

第一轮 13/14，唯一失败项是"第二次请求 cached_tokens 仍为 0"。
但服务端日志明明是 `[hicache rank 5] lookups 2 hits 1` —— 缓存确实命中了。

根因在**我自己的补丁**：非流式路径里 `fe.release()` 在 `finally` 中先执行，
而 `usage()` 的字典在 `try/finally` 之后才构造，于是 `release()` 已经把记录清掉了。
流式路径没这个问题（`usage()` 在生成器里、`finally` 之前求值），所以第一轮流式测试是过的。

修法：`usage()` 接受显式 `cached_tokens` 参数，非流式在 `try` 内先读出计数再进 `finally`。
修完 14/14。

**教训**：不要只看 API 字段判断缓存有没有生效，要拿服务端日志/指标交叉验证——
这次是日志救了判断。

---

## 二、缓存 A/B（仓库自带 `serve/check_prefix_cache.py`）

```
turn 1: prompt 13526 tok | cached_tokens 13312 | TTFT no-cache 1031 ms, cache 338 ms (3.1x) | outputs identical
turn 2: prompt 13546 tok | cached_tokens 13312 | TTFT no-cache 1147 ms, cache 286 ms (4.0x) | outputs identical
turn 3: prompt 13560 tok | cached_tokens 13312 | TTFT no-cache 1097 ms, cache 341 ms (3.2x) | outputs identical
```

服务端对应的日志行（DeepInfer 版比 vLLM 的 `Prefix cache hit rate` 更细）：

```
[hicache rank 3] lookups 6 hits 6 token hit rate 98.3% deferred 0 stores 1 pages 105/75318 states 4/4096 evicted 0
```

要点：
- 命中 **13312 / 13526 token = 98.4%**，TTFT **3.1-4.0 倍**改善；
- **输出逐字相同**（缓存不影响数值结果）；
- `pages 105/75318`、`states 4/4096`、`evicted 0` —— 8 GiB 池子远超当前负载。

---

## 三、性能参数（`tools/serve_bench.py`，全部 `ignore_eos` 生成满长度）

| 场景 | prompt | TTFT p50 | TPOT p50 | 单流 tok/s | 预填 tok/s |
|---|---|---|---|---|---|
| 短 prompt 冷 | 147 | **290 ms** | 26.2 ms | 38.2 | 507 |
| 长 prompt 冷 | 8019 | **910 ms** | 25.9 ms | 38.6 | 8816 |
| 24k 上下文冷 | 24019 | **1937 ms** | 24.5 ms | 40.9 | 12401 |
| 13.5k 上下文**命中缓存** | 13526 | **338 / 286 / 341 ms** | — | — | — |

**并发与聚合吞吐**

| 并发 | max_tokens | TTFT p50 | TTFT p90 | TPOT p50 | 聚合输出 |
|---|---|---|---|---|---|
| 1 | 128 | 270-910 ms | — | 26 ms | 31-37 tok/s |
| 4 | 128 | 1920 ms | **19106 ms** | 32.7 ms | 32.5 tok/s |
| 8 | 128 | 3137 ms | 8489 ms | 74.5 ms | 61.6 tok/s |
| 8 | **512** | 5651 ms | 8440 ms | 30.4 ms | **159.4 tok/s** |

读法：
- **单流稳态约 38-41 tok/s**（TPOT 24-26 ms），这个不随 prompt 长度变化；
- **预填吞吐随长度上升**（507 -> 8816 -> 12401 tok/s），短 prompt 被固定开销主导；
- **批量 8 路稳态 159 tok/s**（3978 tok / 24.95 s）。128-token 那两行被 TTFT 摊薄，不代表吞吐。
- 显存：**124.5 GiB 已用 / 18.7 GiB 空闲**（`--gpu-memory-utilization 0.85`）。

### 两个需要跟进的问题（不建议忽略）

1. **首个请求付 84 s 的 JIT 代价**
   服务路径的 TileLang 内核（`_prepare_verification_metadata`、`_expand_sampling_params`、
   `_fused_sample` ...）在**第一次请求**才编译，8 个 rank 并行编译累计约 84 s。
   第二次请求就降到 9.6 s，重启后（内核缓存已在盘上）进一步降到正常。
   **建议**：服务起来后主动打一个 warmup 请求再接入流量，或把这一步挪进 `server.sh` 的 readiness 判定。

2. **c=4 时 TTFT p90 达 19.1 s**
   p50 只有 1.92 s，p90 却是 19.1 s —— 说明 4 路并发里有请求被饿死在 prefill 调度上
   （`--prefill-any-interval 2` 与 DP/EP 同步的交互）。这是**尾延迟问题**，需要单独定位：
   怀疑多路长 prompt 同时进入时，`max_num_batched_tokens=32768` 被 4×8019 撑满导致 chunk 化，
   且各 rank 必须同步推进，排队的 rank 一直等。建议下一轮用 perfetto/torch profiler 或
   在 `scheduler` 侧打点确认。

---

## 四、本轮改动

| 提交 | 内容 |
|---|---|
| `4abf261` | `serve`：`usage.prompt_tokens_details.cached_tokens`（vLLM 兼容）；A/B 脚本显示该值 |
| `a2377a6` | `serve`：修非流式路径 `release()` 早于 `usage()` 导致 `cached_tokens` 恒为 0 |

新增工具：`tools/serve_test.py`（14 项功能断言）、`tools/serve_bench.py`（TTFT/TPOT/吞吐，支持 `--no-cache` A/B）。
