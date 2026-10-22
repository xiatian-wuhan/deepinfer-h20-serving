# 24 — 消息顺序：和 DeepSeek 官方两份实现对表

现象：经 Claude Code → new-api → 本服务时，怀疑消息前后顺序错乱。

上一轮（[`notes/22`](22-message-order-verification.md)）只证明了"我们这边不重排"。
这一轮换成**用 DeepSeek 自己的实现当尺子**逐项对表，结论分三块：

| # | 结论 |
|---|---|
| 1 | **请求侧完全保序**：本服务渲染出的 prompt 与模型自带的参考编码器**逐 token 相同**（11 组对抗用例），跨轮前缀不变 |
| 2 | **响应侧有一处真实偏差，已修**：DSML 工具块的探测标记写死成 `"\n\n<\|DSML\| calls"`，比官方流式解析器严格，窄条件下会让原始标记漏进正文 |
| 3 | **一处规范差异，未改，记录在案**：官方要求"回答 tool_use 的那条消息必须以 tool_result 开头"，正文在前会被它直接 400；我们和参考编码器都接受 |

---

## 0. 两把"官方尺子"

| 尺子 | 位置 | 用途 |
|---|---|---|
| 离线参考编码器 | `$MODEL/encoding/encoding.py`（990 行，自带 55 个测试） | prompt 渲染 + 输出解析，**本服务的实际依赖** |
| 生产协议框架 | [`deepseek-ai/deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe)（Rust + Python 绑定） | Claude / OpenAI / Responses 三种协议统一转成 Conversation，再渲染 prompt、再把输出流式解码回各协议 |

`deepseek-recipe` 里真正相关的三个文件：

- `deepseek-recipe/src/protocol/anthropic/messages/request/convert.rs` — Claude Messages → Conversation
- `deepseek-recipe-encoding/src/v4/dsv41.rs` — Conversation → V4.1 prompt（含 `normalize_messages`）
- `deepseek-recipe/src/stream/output_parser.rs` — 输出流式解析（"模型解码"那一半）

先看它自带的编码器测试（作为尺子可信度的基线）：

```
docker exec di-e2e sh -c 'cd /model/encoding && python -m pytest -q test_encoding.py'
55 passed
```

---

## 1. 请求侧：三方一致（逐 token 校验）

工具：`tools/check_encoding_parity.py`（在容器里跑，直接调用部署中的 `Frontend.render_prompt`，
和 `encoding.py` 的 `encode_messages` 比 token id）。

用例覆盖最容易出问题的地方：

| 用例 | 形态 | prompt token |
|---|---|---|
| `plain_multiturn` | 纯多轮 | 21 |
| `single_tool` | 单工具 + 结果 + 后续用户文本 | 364 |
| **`parallel_tools_reversed`** | **两个并行调用，网关把结果逆序送来** | 401 |
| `tool_then_text` | 同一 Claude 消息内 `[tool_result, text]` | 360 |
| `text_then_tool` | 同一 Claude 消息内 `[text, tool_result]` | 364 |
| `mid_conversation_system` | 会话中途 system（V4.1 特性） | 20 |
| `interleaved_images` | `[text, image, text]` | 11 |
| `consecutive_user_messages` | 连续两条 user | 11 |
| `no_system_message` | 无 system，只有 user | 306 |
| `assistant_text_plus_calls` | assistant 同时有正文和调用 | 363 |
| `claude_code_like` | 三轮工具循环 | 419 |

结果：**11/11 逐 token 相同**。

逆序那一组的 prompt 实际长这样（说话人都按输入顺序展开）：

```
<<<USER>>> do two things
<<<ASST>>> </think>
<｜DSML｜ calls>
<｜DSML｜ invoke name="lookup"> … </｜DSML｜ invoke>
<｜DSML｜ invoke name="fetch"> … </｜DSML｜ invoke>
</｜DSML｜ calls><｜end▁of▁sentence｜>
<<<USER>>>
<tool_result>A result</tool_result>          ← 网关送来的顺序是 B、A

<tool_result>B result</tool_result>

summarize
<<<ASST>>> </think>
```

`A` 在 `B` 前面，是按 assistant 的 `tool_calls` 顺序重排的结果——这是两把尺子共有的行为，
见下一节。

**跨轮前缀不变**（前缀缓存的前提，也是"模型读到的是同一段历史"的前提）：

```
turn 0 prompt= 309
turn 1 prompt= 348  +39
turn 2 prompt= 362  +14
turn 3 prompt= 401  +39
turn 4 prompt= 413  +12     每一轮都以上一轮为前缀
```

### 两个官方实现为什么一致

| 环节 | `encoding.py` | `deepseek-recipe`（`dsv41.rs`） |
|---|---|---|
| 连续 user 合并 | `merge_tool_messages` 合并 content_blocks | `normalize_messages`：连续 user 用 `"\n\n"` 连接 |
| tool_result 排序 | `sort_tool_results_by_call_order`：只在同一条 user 内、只重排 tool_result 槽位、未匹配 id 排最前 | `sort_tool_results_by_call_order`：只在 user/tool 连续段内、只重排 tool 槽位、未匹配 id 排最前 |
| 一个 user 轮内的分隔 | `"\n\n".join(parts)` | `user_or_separator`：前一条是 User/Tool 就写 `"\n\n"`，否则才写 `<｜User｜>` |

逐行看完再对上第 1 节的实测，"网关把 tool_result 逆序送来"这类情况不会导致乱序。

---

## 2. 修掉的真问题：DSML 探测比官方严

### 现象

旧代码把工具块的起点写死成 `"\n\n<｜DSML｜ calls"`：

```python
self.tool_calls_prefix = f"\n\n<{dsml_token}{tool_calls_block_name}"
start = text.find(self.tool_calls_prefix)
```

只认"两个换行 + 标记"。而官方流式解析器的规则是：

```rust
// output_parser.rs
/// Register the opening marker of the tool-call block, which newlines can precede.
fn add_tool_calls_open_branch(&mut self, dsml_open_marker: &[Segment<'_>]) {
    self.add_branch(MatchPattern::NewlinesAndMarker(pattern(dsml_open_marker)), Stage::ToolCalls, …)
}
```

`NewlinesAndMarker` = **任意个换行（含 0 个）** + `<｜DSML｜`。而且进入 `Stage::ToolCalls` 后
`ActionOnUnmatched::Skip`——标签之间多一个空格、少一个换行都不影响它认。

### 后果

命中不了的形态下，整段 DSML **原样当正文**发给客户端，`tool_calls` 为空。Claude 端看到的是
"模型把 XML 当话说"，工具没被调用，下一轮的工具结果自然对不上——这正是代码注释里
"Claude client printed `<|DSML| calls>`" 那一类故障，只是触发条件更窄。

旧标记在四种写法上的行为（`tools/old_marker_demo.py` 实测）：

| 写法 | 旧代码命中 | 原始标记漏进正文 |
|---|---|---|
| `summary\n\n<｜DSML｜ calls>`（训练分布） | 是 | 否 |
| `summary\n<｜DSML｜ calls>` | **否** | **是** |
| `summary<｜DSML｜ calls>` | **否** | **是** |
| 块在轮首（无正文） | **否** | **是** |
| 标签间带空格 | 是（但严格解析器抛异常） → 仍漏 | 是 |

### 修复（`serve/openai_server.py`）

1. **探测标记**改成 `<｜DSML｜` 本身，再要求紧跟 ` calls`；找不到就继续找下一个候选，
   避免正文里偶尔出现的标记把整段吞掉。
2. **流式 gate** 按"部分标记 + 紧挨着它的换行"回退，命中时把块前导换行从正文里剥掉——
   保证**流式与非流式的 content 完全一致**（旧实现这两者本来一致，改的时候不能破）。
3. **容错解析**：先走官方 `encoding.parse_tool_calls`（严格），抛异常时再用一个
   与 `Stage::ToolCalls` 等价的小扫描器（按字面匹配 `invoke name="` / `parameter name=`，
   跳过标签间空白）。两者都失败才退回"原文照发"。
4. 空块（`<｜DSML｜ calls></｜DSML｜ calls>`）当作"没有工具调用"，把空标记吃掉而不是漏出去。

### 验证

`tools/check_encoding_parity.py` 里 C1/C2 两组，**流式按 2~3 字符切片喂**，与非流式、与
`parse_message_from_completion_text` 三方对比：

```
== C1. 与参考解析器对表 ==
  [ok] text_only / one_call / two_calls / calls_only / arg_has_newline

== C2. 官方能接受、模型不常写的布局 ==
  [ok] no_leading_newlines   （无换行）
  [ok] single_newline        （单换行）
  [ok] block_first           （块在轮首，正文为空）
  [ok] spaced_layout         （标签间带空格，走容错解析）
  [ok] truncated_block       （max_tokens 截断：保留原文，不谎报 tool_calls）
```

> 注意：别用"同一请求跑两次流式/非流式再比文本"来验证顺序。本引擎即使
> `temperature=0` + 固定 `seed` 也不是逐 token 可复现的（MTP 投机解码 + 批组成变化），
> 两次生成的文本会不同，看上去像丢字。要验证切块/顺序，用上面这种确定性用例
> （同一段文本按 2~3 字符切片喂 gate），不要用两次采样。

线上回归：`tools/serve_test.py` **25/25**；`tools/live_order_probe.py` 让模型并行发三个工具调用：

```
content='我会并行执行这三个独立操作。'
[0] read_file({"path": "/etc/hosts"})
[1] run_shell({"cmd": "uname -r"})
[2] web_search({"q": "NVIDIA H20 GPU 显存规格 96GB HBM3"})
DSML leaked into content: False      indices contiguous from 0: True
```

---

## 3. 和官方一致、确认不是 bug 的两处

**块结束之后的文本被丢弃。** `output_parser.rs`：

```rust
Stage::Finished => ActionOnUnmatched::SkipInvalid { kind: InvalidTokenKind::ContentAfterFinished },
```

官方也是丢，并记一条 `ContentAfterFinished`。我们同样丢弃（`split_tool_calls` 取块前文本）。

**截断块的处理不同（有意为之）。** 官方把不完整的块报成不完整；我们把原文照发，
因为服务层报 400 会让客户端连"模型说了什么"都看不到。`finish_reason` 此时仍是 `length`。

---

## 4. 规范差异：回答 tool_use 的那条消息必须以 tool_result 开头

> **后续（[`notes/25`](25-recipe-protocol-layer.md)）**：协议层整体换成 `deepseek-recipe` 之后，
> 这一条不再是我们和官方的差异——本服务现在同样按契约 400。本节保留其原因与影响面的记录。

`convert.rs` 对用户消息逐块处理，文本块前先查"是否还有没被回答的 tool_use"：

```rust
fn ensure_no_pending_tool_uses(…) -> Result<(), ConversionError> { …
    "`tool_use` ids were found without `tool_result` blocks immediately after: {}. Each `tool_use` block \
     must have a corresponding `tool_result` block in the next message." }
```

也就是说 `[text, tool_result]` 这种顺序，官方**直接 400**；只有 `[tool_result, text]` 合法。

我们的链路不会报错：new-api 的 flush 模型保序（`[text, tool_result]` → `user(text)` 然后 `tool`），
`encoding.py` 的 `merge_tool_messages` 再把它们并进同一个 user 轮，渲染成"正文在前、结果在后"。
**两把尺子的渲染层都接受这种形状**（`user_or_separator` 只负责用 `"\n\n"` 连接），
只有 Claude 适配器在转换层拒绝。

**没有改**，理由：

- 合规的 Anthropic 客户端本来就该让 tool_result 紧跟 tool_use，这种形状不该出现；
- 真出现了，把用户正文挪到工具结果之后会改变语义（"先别跑"这类插话会被挪位），
  比"接受并保序渲染"风险大。

如果以后在 `[msgs] in=` 里真看到 `user(...) | tool->...`（正文在前），
在 `render_prompt` 里把紧邻的 `tool` 消息提到该 user 之前即可，一行 `sort`，其余不动。

---

## 5. 复现

```bash
# 尺子自检
docker exec di-e2e sh -c 'cd /model/encoding && python -m pytest -q test_encoding.py'

# 逐 token 对表（容器内，因为要 import 部署中的 openai_server.py）
docker cp tools/check_encoding_parity.py di-e2e:/tmp/
docker exec di-e2e python /tmp/check_encoding_parity.py

# 线上回归
python3 tools/serve_test.py
python3 tools/live_order_probe.py
```

`check_encoding_parity.py` 退出码非 0 即有用例不通过，可以直接挂到发布前检查里。

---

## 6. 真实流量复核（2026-10-22 23:2x，Claude Code 经 new-api）

26 个请求，最长一条 **120 条消息**（39 system / 12 user / 37 assistant / 32 tool）。
用 `tools/analyze_trace.py` 解析服务日志里的 `[msgs] in=` 摘要：

| 检查项 | 结果 |
|---|---|
| 26 轮工具调用，调用 id 与结果 id 是否一一对应 | 26/26 一致，0 不匹配 |
| 结果是否按调用顺序送达 | 26/26 是，0 逆序 |
| 孤立 `tool` 消息（前面没有 `tool_calls`） | 0 |
| 前缀缓存 | 69840 → 70041 → 70345 → 70983 → 71101 → 74021 tokens，cached = 69760 / 70016 / 70272 / 70912 / 71040（**96–99.7%**） |
| DSML 解析失败 | 0 |

**缓存这一列是最硬的证据**：如果历史在轮间被重排，前缀会从重排处断开，`cached` 会塌到很小；
实测一直 ≥96%，说明除新增的尾巴外，每个 token 都与上一轮完全相同——模型读到的就是同一段历史。

### 顺带发现：客户端每轮都带一条 mid-conversation system

顶层 system 之外还有 38 条 system 夹在中间，位置很规律：

| 位置 | 次数 |
|---|---|
| `tool -> system -> assistant` | 25 |
| `user -> system -> assistant` | 12 |
| `tool -> system -> END` | 1 |

不是 new-api 生成的（它只造顶层 system 一条，`convert.rs` 里没有别的 system 创建点），
是 Claude Code 自己发的 system-reminder（49 / 79 字符）。V4.1 的规则是"mid-conversation system
也要接 assistant 头"，所以每个回合的头落在 `<｜System｜>` 之后：

```
<｜User｜>{tool_result}<｜System｜>{reminder}<｜Assistant｜></think>
```

两份官方实现都这么渲染（`render_message` 的 `role == "system" and index > 0` 分支），不是 bug。
若要 A/B，把这类 reminder 并进前一条 user 轮（`<｜User｜>{tool_result}\n\n{reminder}`）即可，
但那是改语义，不建议默认开。
