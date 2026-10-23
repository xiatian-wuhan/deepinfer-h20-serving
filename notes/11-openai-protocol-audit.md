# OpenAI 协议完备性审计（对照 vLLM）

审计对象：`serve/deepseek_server.py`（PR#147 层）+ 权重自带 `encoding/encoding.py`（990 行）
+ 引擎 `SamplingParams` / `ve/multimodal_processor.py`

---

## 一、结论先讲：缺失分三类，容易修的和不容易修的不是同一件事

| 类别 | 含义 | 代表参数 |
|---|---|---|
| **A. 引擎本来支持，服务层没接** | 改服务层几行就能做，**不需要动引擎/内核** | `top_k`、`min_tokens`、`top_logprobs`、`response_format`、`stop_token_ids` |
| **B. 权重自带的编码器支持，服务层把它丢了** | 我们自己的 bug，也不需要动引擎 | **图片输入**、**`reasoning_effort`**、`context`、`drop_thinking` |
| **C. 引擎根本没有这个能力** | 要做必须改采样器/内核，不是协议层能补的 | `presence_penalty`、`frequency_penalty`、`repetition_penalty`、`min_p`、`logit_bias`、`n>1`、`best_of` |

**B 类是最值得先修的**：`encode_messages` 的签名是

```python
def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,
                    add_default_bos_token=True, reasoning_effort=None,
                    return_multi_modal_data=False)
```

而服务层只传了 `thinking_mode=self.args.thinking_mode`（**全局 CLI 参数，不能按请求改**），
`return_multi_modal_data` 没开，所以图片被丢掉。

---

## 二、现状逐项

### 2.1 当前真正生效的参数（只有 5 个）

```python
sampling = dict(temperature=..., top_p=..., ignore_eos=..., max_tokens=...)
if body.get("seed") is not None: sampling["seq_seed"] = int(body["seed"])
```

其余字段**静默忽略**（`body` 是裸 dict，未知 key 不报错）。这对客户端是最危险的：
请求看起来成功了，参数却没生效。

### 2.2 引擎支持但没接的（A 类）

`deepinfer/sampling_params.py:47` 的 `SamplingParams.__init__` 实际接受：

| SamplingParams 字段 | 对应 OpenAI/vLLM 参数 | 现状 |
|---|---|---|
| `top_k` | `top_k`（vLLM 扩展） | ❌ 未接 |
| `first_n_token_no_eos` | `min_tokens` | ❌ 未接（语义正好对应：前 N 个 token 不允许 EOS） |
| `top_logprobs` / `prompt_logprobs` | `logprobs` / `top_logprobs` | ❌ 未接（引擎 `llm_engine.py:747` 确实产出 `output.token_logprobs`） |
| `so_args` | `response_format`（json_schema/json_object） | ❌ 未接 |
| `eos_id` | `stop_token_ids`（单值可映射） | ❌ 未接 |
| `golden_token_ids` | —（内部评测用） | — |

### 2.3 编码器支持但丢了的（B 类）

| 能力 | 编码器签名 | 服务层现状 |
|---|---|---|
| 图片 | `return_multi_modal_data=True` → 返回 `(prompt, {"images": [...]})` | ❌ 未开；worker 里 `mm_data()` **硬编码** `dict(image_bytes=[], image_sizes=[])` |
| `reasoning_effort` | 支持 `"low"/"medium"/"high"` 或 **数值 1-100**，仅在 thinking 模式下渲染 | ❌ 未传 |
| `thinking_mode` | 每请求可传 | ⚠️ 只有全局 `--thinking-mode`，不能按请求切 |
| `context` / `drop_thinking` | 支持 | ❌ 未接 |

**多模态的打通路径已经查清，不需要动引擎：**

```
编码器产出： {"images": [{"type":"image","url": <路径 | data: | http(s)>}, ...]}   # 按占位符出现顺序
引擎契约：   multi_modal_data = {"image_sizes": [(w,h), ...], "image_bytes": [bytes|None, ...]}
```
（`ve/multimodal_processor.py:112` 原文："image_sizes 按图像占位 token 出现顺序提供每图 (width, height)；
image_bytes 为编码后的图像字节，可缺省"）

所以服务层需要做：解析 `image_url` 内容块 → 编码器收集 → 对每个记录解析 url（data URI / 本地路径 / http）
→ 用 PIL 取 (w,h) → 组装 `image_sizes`/`image_bytes` → 透传到 worker 的 `mm_data()`。

**惩罚项（C 类）已确认引擎不支持**：全仓库 grep `repetition_penalty|presence_penalty|frequency_penalty|min_p`
只在 `min_pixels`（视觉，无关）里命中。要支持必须改采样器。

### 2.4 错误返回

现状只有两处非标准错误：

```python
return JSONResponse({"error": {"message": "engine loading"}}, status_code=503)
return JSONResponse({"error": {"message": f"prompt has {n} tokens > max_model_len"}}, status_code=400)
```

问题：
1. 缺 OpenAI 标准四字段 `type` / `param` / `code`；
2. **没有任何参数校验**（非法 `temperature`、负数 `max_tokens`、未知字段都不会报错）；
3. body 不是合法 JSON 时，FastAPI 会吐自己的 422 结构，不是 OpenAI 格式；
4. 参数不支持时不拒绝——静默忽略比报错更糟。

### 2.5 缓存命中信息

已实现（本轮补的）：`usage.prompt_tokens_details.cached_tokens`，流式非流式都有。
更细的命中细节（pages/states/deferred/evicted）目前只有 `/deepinfer/metrics` 的**按 rank 聚合**值，
**没有按请求**返回。vLLM 也只返回 `cached_tokens`，所以标准层已经对齐；
若要按请求给更细的，需要新增非标准字段（建议放到响应体的 `deepinfer` 扩展块，需你确认是否要）。

---

## 三、实施计划

### 第一批（纯服务层，零引擎改动，风险低）
1. OpenAI 标准错误封装 + 参数白名单校验：
   非法值 → `400 invalid_request_error`（带 `param`）；C 类参数 → `400 unsupported_parameter` 并说明原因。
2. 接 `top_k`、`min_tokens`→`first_n_token_no_eos`、`response_format`→`so_args`、`stop_token_ids`→`eos_id`。
3. `reasoning_effort` + 按请求 `thinking`：透传给 `encode_messages`。
4. `stop`（字符串）：服务层在流式/非流式输出上做截断（引擎无多 stop 支持）。
5. 图片输入：`return_multi_modal_data=True` + url 解析 + PIL 取尺寸 + 透传到 `mm_data()`。
6. `n>1`：服务层扇出 N 个请求（引擎支持多序列）。

### 第二批（需要引擎/内核）
7. `presence_penalty` / `frequency_penalty` / `repetition_penalty` / `min_p`：采样器加 logits 后处理。
8. `logit_bias`：同上。

### 需要你拍板
- 惩罚项是否要动采样器？（会影响 `_fused_sample` 内核的签名/性能）
- 是否接受非标准的 `deepinfer` 扩展块来放按请求的缓存细节？

---

## 四、参考：vLLM 对应实现位置

- `vllm/entrypoints/openai/protocol.py` — `ChatCompletionRequest`（参数全集与校验）
- `vllm/entrypoints/openai/serving_chat.py` — 参数到 `SamplingParams` 的映射
- `vllm/entrypoints/openai/serving_engine.py` — 错误封装 `create_error_response`
- `vllm/entrypoints/openai/protocol.py::UsageInfo` — `prompt_tokens_details.cached_tokens`
