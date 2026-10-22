# 25 — 协议层换成 deepseek-recipe：把"协议"还给协议框架

[`notes/24`](24-message-order-vs-official-encoding.md) 的结论是"我们和官方两份实现对得上"，
代价是**同一套规则我们维护了两份实现**（`serve/openai_server.py` 里手写的渲染/解析，
对照 `$MODEL/encoding/encoding.py`），并且已经为此付过三次修复：DSML 探测标记过严、
流式门控漏换行、容错解析器差一。

这一轮换掉前提：**协议转换不该由服务层重写，应该直接用 OpenAI 放出来的那份实现。**

| # | 结论 |
|---|---|
| 1 | `deepseek-recipe` 已经以 wheel 的形式**装在镜像里**（`import deepseek_recipe`，0.2.0），不需要新增依赖 |
| 2 | 请求侧：11 组对抗用例里 10 组**逐 token 相同**，第 11 组是它按 OpenAI 契约拒绝（见 §4.1） |
| 3 | 响应侧：5 组 DSML 输出，流式/非流式解析结果与参考解析器完全一致 |
| 4 | 换完**多**出来：`/v1/messages`、`/v1/responses`、`reasoning_content`、`reasoning_tokens`、`prompt_cache_hit_tokens`、控制符注入防护 |
| 5 | `--protocol-layer builtin` 保留旧路径做 A/B（默认 `recipe`） |

---

## 0. 第一性原理：这条边界该划在哪

一次请求要过四道手：

```
HTTP 报文 → ① 协议 → Conversation   → ② Conversation → prompt/prompt_ids
            ↑ 各协议各不相同           ↑ 模型既定格式，模型变了才变
prompt_ids → ③ 引擎 → 采样出的 token_ids → ④ token_ids → 各协议流式响应
```

- ①④ 是**协议知识**：Chat Completions / Messages / Responses 各自的字段、默认值、错误码、
  SSE 事件名、usage 字段名。这些不由我们定义，也不该由我们猜。
- ② 是**模型知识**：`<｜User｜>` / `</think>` / DSML 块长什么样。checkpoint 自带
  `encoding/encoding.py`，官方还在持续改（V4 和 V4.1 的模板就不一样）。
- ③ 是**我们唯一的领域**：8 卡 CP8×EP8、前缀缓存、YoCo、MTP。

所以边界应该划在 **token_ids**：引擎只认 token id、只吐 token id，前面全部交给协议框架。
我们过去把 ①②④ 都写在了 `openai_server.py` 里——那不是"服务层薄"，是把两份外部规格
抄进了自己的代码，然后靠 `tools/check_encoding_parity.py` 反向验证抄得对不对。

`deepseek-recipe` 正是官方为此发布的库，README 第一句就是这个分工：

> Use these components to connect an inference backend to API services that support
> multiple formats. **The caller must provide model inference, tool execution, and HTTP
> transport.**

即：它管 ①②④，我们只要实现"给定 prompt token ids，吐出 token ids"的 ③。

---

## 1. 对表：换掉之前先证明换掉不改字节

工具 [`tools/recipe_parity.py`](../tools/recipe_parity.py)，两组尺子：

- 真值：`$MODEL/encoding/encoding.py`（本服务此前的实际依赖）
- 被测：`deepseek_recipe`

### A. 请求侧（prompt 文本 + token ids）

| 用例 | token 数 | 结果 |
|---|---|---|
| `plain_multiturn` | 21 | ✅ 逐 token 相同 |
| `single_tool` | 364 | ✅ |
| `parallel_tools_reversed` | 401 | ✅ 结果乱序送达也按调用序重排 |
| `tool_then_text` | 360 | ✅ |
| `text_then_tool` | — | ⛔ 被拒（§4.1） |
| `mid_conversation_system` | 20 | ✅ |
| `interleaved_images` | 11 | ✅ 见 §4.3 |
| `consecutive_user_messages` | 11 | ✅ |
| `no_system_message` | 306 | ✅ |
| `assistant_text_plus_calls` | 363 | ✅ |
| `claude_code_like` | 419 | ✅ 三轮回合 |

`10/10 一致；1 组被 OpenAI 契约拒绝`。

关键点：**token ids 相同**，不只是文本相同。前缀缓存按 token 页做内容寻址，
文本一致而 token 不一致等于缓存全废；这条是真检查。

### B. 控制符注入

用户消息里写 `<｜Assistant｜>` 这几个字：

| 路径 | 该控制符出现次数 |
|---|---|
| 旧路径（`tokenizer(prompt, add_special_tokens=False)`） | **2**（用户那条变成了真的 assistant 头） |
| `deepseek_recipe` 的 prompt encoding | **1**（只有真正的那个头） |

官方编码器把"渲染期写入的控制符"和"用户文本"分开处理：plain 区间按**去掉 added vocab**
的 tokenizer 编码，所以用户手打的控制符只是普通文字。这是一处真实的安全修复，不是洁癖。

### C. 响应侧（DSML → tool_calls）

同一段模型输出，分别喂给本服务与参考解析器，比对 `content` / `tool_calls` / `finish_reason`：

| 输出 | 调用数 | finish_reason | completion_tokens |
|---|---|---|---|
| `text_only` | 0 | stop | 4 |
| `one_call` | 1 | tool_calls | 75 |
| `two_calls` | 2 | tool_calls | 92 |
| `calls_only` | 1 | tool_calls | 72 |
| `arg_has_newline` | 1 | stop→tool_calls | 52 |

流式与非流式两遍都完全一致。`Stage::Finished => ContentAfterFinished` 之后的多余文本被丢弃，
和 `notes/24` §3 记录的官方行为一致。

---

## 2. 换完之后多出来的东西

这些不是"顺带"，是把手写层删掉换来的：

| 能力 | 旧路径 | 现在 |
|---|---|---|
| `/v1/messages`（Anthropic） | 无 | ✅ 原生，含 `message_start`/`content_block_delta`/`message_delta` 事件与 `cache_read_input_tokens` |
| `/v1/responses`（OpenAI Responses） | 无 | ✅ 原生，含 `output[].content[].output_text` |
| 思考模式输出切分 | 无（`<think>` 会漏进正文） | ✅ `delta.reasoning_content` + `completion_tokens_details.reasoning_tokens` |
| 缓存命中 | 手写 `prompt_tokens_details.cached_tokens` | ✅ 三协议各自的字段：`cached_tokens`、`prompt_cache_hit_tokens`/`miss`、`cache_read_input_tokens` |
| 用户文本里的控制符 | 会变成真控制符 | ✅ 只是文字 |
| DSML 近失配容错 | 手写 `tolerant_tool_calls`（曾差一） | ✅ 官方解析器的 `NewlinesAndMarker` + `Skip` |

思考模式实测（`reasoning_effort: low`）：

```json
"usage": {"prompt_tokens": 37, "completion_tokens": 18,
          "completion_tokens_details": {"reasoning_tokens": 16}}
"message": {"reasoning_content": "…49 字…", "content": "391"}
```

---

## 3. 服务层现在长什么样

```
openai_server.py
├── worker_main           引擎侧（CP8×EP8、前缀缓存、YoCo）—— 未改
├── Frontend
│   ├── submit/release    rid ↔ DP rank、缓存命中数、usage 记账 —— 未改
│   ├── _init_builtin     旧路径：checkpoint encoder + 手写解析（--protocol-layer builtin）
│   ├── _init_recipe      ConversationOptions / Encoding / StreamProcessor（默认）
│   ├── prepare()         协议 → Conversation → prompt_ids（CPU，丢线程池）
│   └── sampling()        temperature/top_p/top_k/seed → SamplingParams
├── recipe_response()     引擎 token_ids → StreamProcessor → 各协议 SSE/JSON
└── build_app()           /health /v1/models /deepinfer/metrics + 三个协议端点
```

一次请求的路径：

1. `prepare()`：`ChatCompletionRequest(body).convert(options)` → `DeepseekV41Encoding().encode()`
   → `prompt_ids`，同时拿到 `parsing_options`（有没有 tools、停用词、思考阶段）。
2. `submit()`：照旧进队列，引擎不知道协议层换了。
3. `recipe_response()` 的首个 token 之前先 `InferenceChunk.ready(prompt_usage=PromptUsage(
   prompt_tokens, prompt_cache_hit_tokens))`——缓存命中数在 worker 里先于 token 入队，此时已定。
4. 每个 token → `InferenceChunk.tokens([id])` → `StreamProcessor.push()` → 协议 chunk → `data: …`。
5. 结束 → `finish(Length|Stop)`（有工具调用时解析器自己改写成 `tool_calls`）→ `processor.finish()`。

---

## 4. 三处有意的行为变化

### 4.1 `text_then_tool` 从"渲染"改成"400"

一条 assistant 的 `tool_calls` 后面，先来一条 `user`、再来 `tool` 结果：

```
assistant(tool_calls) → user("题外话") → tool(result)
```

- 旧路径：照渲染，模型看到"题外话"再看到工具结果。
- `deepseek_recipe`：`ensure_no_pending_tool_uses` 直接 400 ——
  "an assistant message with 'tool_calls' must be followed by tool messages…"，这是 OpenAI
  自己的契约（`notes/24` §4 记录过，官方生产框架一直这么拒）。

真实流量里没出现过：`notes/24` §6 的 26 轮 Claude Code 轨迹里 0 次。new-api 只有在一条
Claude 消息把 text 块放在 tool_result 块前面时才会这么拼。所以按契约拒掉，不保留宽容。

### 4.2 `top_p` / `top_k` / penalty 仍然从报文取

`ConversionOptions` 也产出 `InferenceOptions`，但：

- V4.1 **非思考模式**下它把 `top_p` 硬钉成 `1.0`（`ModelVersion::resolve_top_p`），
- `top_k` 不在 `InferenceOptions` 里，
- `presence_penalty` / `frequency_penalty` 校验后**丢弃**（DeepSeek 官方 API 不支持）。

所以采样参数照旧从原始报文取，直接给引擎的 `SamplingParams`；`recipe` 只负责
`max_tokens` 的解析（`resolve_max_tokens`：非思考默认 8192、思考 65536、`max` 努力 131072，
并按上下文剩余量下调）。客户端设了 `top_p=0.5` 就该是 0.5——静默改成 1.0 比不实现更糟。

### 4.3 图片占位符的拼写

checkpoint 里 `image_token_id = 129264`，但**表面拼写**是 `<｜deepseek_image｜>`；
官方发布的 V4.1 tokenizer 同一个 id 拼成 `<｜image｜>`。id 才是模型看到的，拼写只是 tokenizer 的事实。

处理方式：`ConversionOptions().with_special_tokens(SpecialTokens(image=…))`，
拼写从 `/model/tokenizer.json` 的 `added_tokens` 里按 id 反查（`_checkpoint_image_token`）。
不做这件事的后果是静默的——占位符会被当普通文字编码，图片根本到不了视觉编码器。

另外 `ImageInfo` 的像素预处理是 recipe 自己的一套，引擎要的是原始字节 + 自己的
`multi_modal_processor`，所以图片**取字节和算 token 数仍走本地逻辑**（`_resolve_images`），
recipe 只负责在 prompt 里按顺序插占位符。

顺带一处收紧：Messages 形状的图片块（`{"type":"image","source":…}`）发到
`/v1/chat/completions` 现在 400（OpenAI schema 只认 `image_url`），要发就发 `/v1/messages`。

---

## 5. 启动与开关

```bash
# 默认：deepseek-recipe 协议层，三个端点
serve/server.sh start --dp 8 --cp 8 --max-num-seqs 32 --port 8888 --log-requests

# A/B 旧路径
serve/server.sh start --dp 8 --cp 8 --max-num-seqs 32 --port 8888 --protocol-layer builtin
```

启动日志会打印实际生效的层：

```
[server] protocol layer: deepseek-recipe 0.2.0 (image token <｜deepseek_image｜>, eos 1)
```

---

## 6. 验证

```
$ docker exec di-e2e python /tmp/recipe_parity.py
10/10 cases prompt- and token-identical; 1 refused by the OpenAI contract (text_then_tool)
[ok] A3 用户文本里的控制符只是文字
[ok] A4 图片占位符与 ids 一致
all checks passed

$ docker exec di-e2e env SERVE_TEST_IMAGE=/tmp/test.jpg python /tmp/serve_test.py
===== 结果: 26 passed, 0 failed =====

$ docker exec di-e2e python /tmp/live_order_probe.py
content='我会并行执行这三个操作。'  3 个 tool_calls，索引连续，无 DSML 泄漏
```

`serve_test.py` 里新增的两条断言固化了 §4.3 的收紧：
`Anthropic image 块可用（/v1/messages）` 与 `chat 端点拒绝 Anthropic image 块`。

图片实测：768×768 JPEG，`usage.prompt_tokens = 393`（纯文本对照 9），
差值即视觉编码器展开的 token 数，说明占位符确实落到了 `image_token_id` 上。

---

## 7. 复现

```bash
# 1) 协议层对表（不需要 GPU，任意装了 deepseek-recipe 的容器）
docker run -d --name dr-spike -v /path/to/model:/model:ro \
    -v "$PWD/serve":/opt/deepinfer/DeepInfer/serve:ro deepinfer-sm90:v41-sm90 sleep infinity
docker cp tools/check_encoding_parity.py dr-spike:/tmp/
docker cp tools/recipe_parity.py dr-spike:/tmp/
docker exec -i dr-spike python /tmp/recipe_parity.py

# 2) 在线套件（需要引擎）
docker cp tools/serve_test.py tools/live_order_probe.py di-e2e:/tmp/
docker exec di-e2e python /tmp/serve_test.py
docker exec di-e2e python /tmp/live_order_probe.py
```

> 注：`deepseek_recipe` 已随镜像安装（`/opt/deepinfer/venv`），换个容器不需要重装；
> 若在别的基础镜像上重建，加一条 `pip install deepseek-recipe==0.2.0` 即可（cp310-abi3 wheel）。
