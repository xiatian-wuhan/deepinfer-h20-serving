# 接口参考

生产服务在 **`http://172.100.52.238:8888`**（容器内是 `127.0.0.1:8888`）。三套会话协议共用同一台引擎、
同一份前缀缓存，请求体都是 JSON，流式都是 SSE。

- **无鉴权**：内网部署，`Authorization` 头可以不发，发了也不校验。
- **模型名**：`DeepSeek-V4.1-Flash-1017`。也接受 `DeepSeek-V4.1-Flash`、`DeepSeek-V4-Pro` 和 `claude-*`
  （DeepSeek 官网的 Anthropic 面就这么映射），但本机只有这一个模型，响应的 `model` 字段一律回
  `DeepSeek-V4.1-Flash-1017`。
- **自动生成的文档**：`GET /docs`（Swagger UI）和 `GET /openapi.json` 只有路由列表——请求体是裸 JSON，
  没有 pydantic schema，所以那里看不到字段。字段以本文为准。

## 路由

| 路径 | 协议 | 说明 |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions | 主力端点：文本、图片、工具、思考、JSON、前缀续写 |
| `POST /v1/responses` | OpenAI Responses | 同一套能力，字段按 Responses 规范 |
| `POST /v1/messages` | Anthropic Messages | Claude 系客户端用 |
| `POST /anthropic/v1/messages` | Anthropic Messages | 与上面等价，官网 `base_url=.../anthropic` 的习惯 |
| `POST /v1/messages/count_tokens` | Anthropic | 只算 token，不生成 |
| `POST /anthropic/v1/messages/count_tokens` | Anthropic | 同上 |
| `GET /v1/models` | OpenAI | 返回本机模型 |
| `GET /health` | — | `{"status":"ok","ready_ranks":1}`；未就绪时 503 |
| `GET /deepinfer/metrics` | Prometheus | 请求数、token 数、每 rank 负载、前缀缓存计数 |

同一句话，三种写法：

```bash
BASE=http://172.100.52.238:8888

# Chat Completions
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'

# Responses
curl -s $BASE/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","input":"你好","max_output_tokens":64}'

# Messages
curl -s $BASE/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","max_tokens":64,"messages":[{"role":"user","content":"你好"}]}'
```

## Chat Completions

### 非流式

```bash
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-1017",
  "messages": [{"role": "system", "content": "你是简洁助手"},
               {"role": "user", "content": "用一句话说明什么是 KV cache"}],
  "max_tokens": 128,
  "temperature": 0.6,
  "top_p": 0.95
}'
```

返回标准 `chat.completion` 对象，`usage` 里带缓存拆分：

```json
{
  "id": "chat-…", "object": "chat.completion", "created": 1792718512,
  "model": "DeepSeek-V4.1-Flash-1017",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}],
  "usage": {
    "prompt_tokens": 2818, "completion_tokens": 42, "total_tokens": 2860,
    "prompt_tokens_details": {"cached_tokens": 2560},
    "prompt_cache_hit_tokens": 2560, "prompt_cache_miss_tokens": 258
  }
}
```

`prompt_tokens_details.cached_tokens` 是 OpenAI 字段，`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
是 DeepSeek 官网同名字段，两者同时给，数值相同。

### 流式

`stream: true` 之后是 SSE：

```
data: {"id":"chat-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"usage":null}
data: {"id":"chat-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}],"usage":null}
…
data: {"id":"chat-…","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":2818,…}}
data: [DONE]
```

两个要注意的点：

- **最后一个 chunk 的 `choices` 是空数组**，只有 `usage`。直接取 `choices[0]` 的客户端要判空。
- **不带 `stream_options.include_usage` 也会发这个 usage 块**。这是刻意的：new-api / one-api
  这类网关只从上游 usage 取计费和缓存数字，丢了它就退化成网关自己分词、缓存永远显示 0。

### 思考模式

```json
{"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
```

返回里思考内容在 `message.reasoning_content`（与 `content` 并列），`usage.completion_tokens_details.reasoning_tokens`
给出思考消耗：

```json
{"message": {"role": "assistant", "content": "2", "reasoning_content": "We need answer only digit…"},
 "usage": {"completion_tokens_details": {"reasoning_tokens": 33}}}
```

默认**不开**思考（服务以 `--thinking-mode chat` 启动）：不传 `thinking` 就是非思考模式。
`reasoning_effort`（`low` / `medium` / `high`）在思考模式下选努力档。

### 工具调用

```json
{
  "model": "DeepSeek-V4.1-Flash-1017",
  "messages": [{"role": "user", "content": "列出 /tmp 下的文件"}],
  "max_tokens": 200,
  "tools": [{"type": "function", "function": {
    "name": "Bash", "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
}
```

```json
{"choices": [{"finish_reason": "tool_calls",
              "message": {"content": "",
                          "tool_calls": [{"index": 0, "id": "call_…", "type": "function",
                                          "function": {"name": "Bash", "arguments": "{\"command\": \"ls -la /tmp\"}"}}]}}]}
```

- `tool_choice`: `auto`（默认）/ `none` / `required` / 指定某个函数。
- 流式时按 OpenAI 约定发 `delta.tool_calls`（先 `name` 后 `arguments`），`finish_reason` 单独一个 chunk。
- 模型写出的 DSML 块由 deepseek-recipe 的官方解析器转成结构化 `tool_calls`，不会漏进 `content`；
  `max_tokens` 截断导致块不完整时原文照发，便于排查。

### JSON 输出

```json
{"response_format": {"type": "json_object"}}
{"response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": {
  "type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}}}
```

schema 会被写进 prompt 约束输出（不是 logits 级别的约束解码），实测返回 `{"a": 42}`。

### 前缀续写

assistant 消息上加 `"prefix": true`，模型从这段前缀继续写，**只回续写部分，不回显前缀**：

```json
{"messages": [{"role": "user", "content": "1, 2, 3,"},
              {"role": "assistant", "content": " 4, 5,", "prefix": true}],
 "max_tokens": 16}
```

### 图片

`content` 用 OpenAI 的 `image_url` 形状，**只接受内联 base64**（服务无鉴权，放开任意 URL 等于开 SSRF 代理；
需要时启动加 `--allow-image-urls`）：

```bash
B64=$(base64 -w0 your.jpg)
curl -s $BASE/v1/chat/completions -H 'Content-Type: application/json' -d "{
  \"model\": \"DeepSeek-V4.1-Flash-1017\",
  \"messages\": [{\"role\": \"user\", \"content\": [
     {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$B64\"}},
     {\"type\": \"text\", \"text\": \"这张图里是什么？\"}]}],
  \"max_tokens\": 256}"
```

- 每张图在 prompt 里落一个 `<｜deepseek_image｜>`，由引擎展开成视觉 token 网格；`usage.prompt_tokens`
  **计入展开后的 token**（768×768 ≈ 393 token）。
- 单请求可多图且保持顺序；图片也可以落在 `tool_result` 里。
- 单图上限 64 MiB / `max_image_tokens=1024` 个 prompt token；解码失败直接 400，不静默降级为纯文本。
- 形状跟着端点走：chat 端点只收 `image_url`，Anthropic 的 `image`+`source` 形状发到 chat 会 400
  （OpenAI schema 如此），发到 `/v1/messages` 才对。

### 参数

| 参数 | 状态 |
|---|---|
| `model` `messages` `max_tokens` `max_completion_tokens` | 生效 |
| `temperature` `top_p` `top_k` `seed` `stop` | 生效 |
| `stream` `stream_options.include_usage` | 生效 |
| `tools` `tool_choice` | 生效（双向：渲染进 prompt，输出解析回 `tool_calls`） |
| `response_format`（`json_object` / `json_schema`） | 生效（写进 prompt 约束） |
| `thinking` `reasoning_effort` | 生效（切思考模式 / 努力档） |
| `prefix`（assistant 消息上） | 生效 |
| `ignore_eos` | 非标准，压测用：强制生成满 `max_tokens` |
| `prefix_cache` | 非标准，默认 `true`；`false` 跳过缓存查询做 A/B |
| `presence_penalty` `frequency_penalty` | 接受但忽略（DeepSeek 官方 API 也是这个语义） |
| `logprobs` `top_logprobs` `min_tokens` `logit_bias` `stop_token_ids` | 接受但不产出 |
| `n > 1` | **400**（引擎只做单条） |
| `max_tokens` 超出剩余上下文 | **400** |
| 未知模型名 | **400** |

采样参数（`temperature`/`top_p`/`top_k`/`seed`/`ignore_eos`）**从请求体原样取**，不走 recipe 的
V4.1 默认值——非思考模式下官方会把 `top_p` 钉到 1.0，那会让调用方设的值失效。

### stop / stop_sequences

`stop`（chat）与 `stop_sequences`（messages）由**协议层解析，不是引擎参数**：引擎的 `SamplingParams`
没有 `stop` 字段，所以它照常往 EOS / `max_tokens` 解码，由 Rust 解析器盯着解出的文本做截断。

命中后的行为：

| 项 | 值 |
|---|---|
| 正文 | 只到命中点之前，**不含** stop 串；命中点之后的尾巴一律不返回 |
| 结束原因 | `stop_reason: "stop_sequence"`，并在 `message_delta.stop_sequence` 回带命中的串 |
| `usage` | 截断在命中点，按实际交付的 token 计 |
| 引擎侧 | 立即 abort 该序列，否则它会继续解码到 EOS，白占一个 KV slot |
| 指标 | `/deepinfer/metrics` 的 `deepinfer:stop_sequences_total` +1，`errors_total` **不**加 |

2026-10-23 之前这条路径有缺陷：解析器收尾后前端还在推 token，下一次 `push` 抛
`RuntimeError: stream processor has finished`。非流式请求会变成纯文本 `500`，流式则因为
`message_stop` 已经发出而表现为**静默截断**（客户端拿到半截正文却以为正常结束）。
回归用例见 [`docker/verify_stop_sequence.py`](../docker/verify_stop_sequence.py)。

## Responses

```bash
curl -s $BASE/v1/responses -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-1017",
  "instructions": "只用中文",
  "input": [{"role": "user", "content": [{"type": "input_text", "text": "说两个字：收到"}]}],
  "max_output_tokens": 64
}'
```

`input` 可以是字符串，也可以是 message 数组。返回对象的关键字段：

```json
{"id": "responses-…", "object": "response", "status": "completed",
 "model": "DeepSeek-V4.1-Flash-1017",
 "output": [{"type": "message", "id": "msg_…", "status": "completed", "role": "assistant",
             "phase": "final_answer",
             "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "收到"}]}],
 "usage": {"input_tokens": 8, "input_tokens_details": {"cached_tokens": 0},
           "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 9},
 "incomplete_details": null}
```

`status` 是 `completed` 或 `incomplete`（命中输出上限时，`incomplete_details` 给原因）。
缓存命中看 `usage.input_tokens_details.cached_tokens`——和官网 Responses 用的是同一个字段。

流式事件（`stream: true`）：

```
response.created → response.in_progress → response.output_item.added
→ response.content_part.added → response.output_text.delta × N
→ response.output_text.done → response.content_part.done → response.output_item.done
→ response.completed（或 response.incomplete）
```

## Messages（Anthropic）

```bash
curl -s $BASE/v1/messages -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-1017",
  "max_tokens": 256,
  "system": "你是简洁助手",
  "thinking": {"type": "enabled", "budget_tokens": 64},
  "messages": [{"role": "user", "content": "1+1 只给数字"}]
}'
```

```json
{"id": "messages-…", "type": "message", "role": "assistant",
 "model": "DeepSeek-V4.1-Flash-1017",
 "content": [{"type": "thinking", "thinking": "…", "signature": "messages-…"},
             {"type": "text", "text": "2"}],
 "stop_reason": "end_turn", "stop_sequence": null,
 "usage": {"input_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
           "output_tokens": 42, "service_tier": "standard"}}
```

**`input_tokens` 不含缓存命中部分**，这是 Anthropic 自己的语义：总输入 =
`input_tokens + cache_read_input_tokens`。`cache_creation_input_tokens` 恒为 0——缓存是自动的，
没有显式 `cache_control`（官网在这个面上同样忽略 `cache_control`）。

流式事件（标准 Anthropic 事件流）：

```
message_start → content_block_start → content_block_delta × N → content_block_stop
→ message_delta → message_stop     （外加 ping）
```

图片用 Anthropic 形状（`{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":…}}`）。

### count_tokens

```bash
curl -s $BASE/v1/messages/count_tokens -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"用一句话说明什么是 KV cache。"}]}'
# {"input_tokens": 11}
```

与同一请求的 `usage.input_tokens` 一致（图片按展开后的 token 计）。

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://172.100.52.238:8888/anthropic
export ANTHROPIC_API_KEY=x        # 随便非空，服务不校验
```

`claude-sonnet-*` / `claude-opus-*` 这类模型名会被映射到本机模型，响应里回的是
`DeepSeek-V4.1-Flash-1017`。

## 错误

统一是 OpenAI 形状的错误对象（Anthropic 端点也一样，这是 deepseek-recipe 的行为）：

```json
{"error": {"message": "…", "type": "invalid_request_error", "param": null, "code": "invalid_request_error"}}
```

| 场景 | 状态码 | message 示例 |
|---|---|---|
| 未知模型名 | 400 | `The supported API model names are DeepSeek-V4-Pro, DeepSeek-V4.1-Flash, DeepSeek-V4.1-Flash-1017, but you passed gpt-4o.` |
| `n > 1` | 400 | `Invalid n value (currently only n = 1 is supported)` |
| 请求体缺字段 | 400 | `invalid messages request: invalid request body: missing field \`messages\`` |
| 请求体不是合法 JSON | 400 | `invalid JSON body: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)` |
| 请求体不是 JSON 对象 | 400 | `request body must be a JSON object, got list` |
| 图片块形状不符 / 解码失败 | 400 | `image block has no data/source/url` 等 |
| `max_tokens` 超出剩余上下文 | 400 | `prompt has N tokens > max_model_len` |
| 引擎未就绪 | 503 | `{"error":{"message":"engine loading"}}` |

2026-10-23 之前，`await request.json()` 抛的 `JSONDecodeError` 没有任何 handler 接管：FastAPI 不
认识它，Starlette 兜底成纯文本 `500 Internal Server Error`，异常只落在 `serve.log`。现在四个入口
统一经过 `_json_body()` + 一个 app 级 handler，非 JSON 对象也一并拦在边界上；回归用例见
[`docker/verify_request_errors.py`](../docker/verify_request_errors.py)。

## 观测

```bash
curl -s $BASE/health              # {"status":"ok","ready_ranks":1}
curl -s $BASE/v1/models
curl -s $BASE/deepinfer/metrics   # Prometheus 文本
```

`/deepinfer/metrics` 的指标族：`deepinfer:requests_total`、`finished_total`、`aborted_total`、
`errors_total`、`prompt_tokens_total`、`completion_tokens_total`、`encode_fallbacks_total`、
`num_requests_running{dp_rank}`，以及每 rank 的 `deepinfer_hicache:lookups|hits|hit_tokens|lookup_tokens`。

服务端日志（`--log-requests`）每个请求一行：

```
[req] rid=0ff79fea stream=1 include_usage=1 images=0 prompt=31565 cached=31104
```

排障时先看这一行：`cached=0` 说明真的没命中（前缀短于 128 token、或服务刚重启缓存是空的）。

## 并发

`--max-num-seqs 32` 是整机的并发上限（CP8 只有一个引擎）。1M 上下文的长请求会把 8 张卡绑成
同一个 batch，建议在网关侧对长请求单独限流；布局与并发的取舍见 [`README §5.4`](../README.md#54-布局怎么选)。

## 相关文档

- [`README`](../README.md) —— 这套服务的全貌：布局选型、性能数字、已知限制
- [`README §6.6`](../README.md#66-和-deepseek-官网接口的对照) —— 和 DeepSeek 官网接口的逐项对照（还差哪些）
- [`notes/25`](../notes/25-recipe-protocol-layer.md) —— 协议层为什么交给 deepseek-recipe
- [`data/`](../data/README.md) —— 性能数字的原始出处与复跑方法
