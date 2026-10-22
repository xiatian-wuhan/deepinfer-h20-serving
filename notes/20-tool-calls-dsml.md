# 20 — 工具调用：把 DSML 解析回结构化 tool_calls

现象：Claude Code（Anthropic 客户端）经 new-api 调用本服务时，模型输出被当成普通文本：
出现 `<｜DSML｜ calls>` / `<｜DSML｜ invoke name="Bash">` 之类的裸标记，工具不会被执行。

## 责任判定：本服务的问题

用带 `tools` 的请求直接打服务，改动前的响应是：

```
finish_reason: stop
tool_calls 字段: 不存在
content: '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="Bash">\n...'
```

即：请求侧我们通过模型自带的 `encoding.encode_messages` 把 `tools` 渲染进 prompt，
模型也正确按格式输出了 DSML，但**响应侧从来没有解析**，直接把标记当文本返回。

new-api 那侧没有问题：它把上游 OpenAI 的 `delta.tool_calls` 转成 Anthropic 的
`content_block_start: tool_use` + `input_json_delta`
（`relaykit/relayconvert/internal/oai_chat/to_claude_messages_resp.go`）。
上游不给结构化字段，它只能原样透传。

## 改法

用模型自带的同一个参考实现解析（`/model/encoding/encoding.py`）：

```python
prefix = f"\n\n<{dsml_token}{tool_calls_block_name}"      # "\n\n<｜DSML｜ calls"
_, _, calls = parse_tool_calls(pos + len(prefix), text)   # -> [{'name','arguments'}]
```

- **非流式**：`message.tool_calls = [{id, type:"function", function:{name, arguments}}]`，
  `finish_reason = "tool_calls"`，`content` 为块前文本（可能为空串）。
- **流式**：用 `_ToolCallGate` 只扣留"与起始标记等长"的尾巴，检测到标记后停止外发文本、
  缓冲整块；结束时按 OpenAI 约定发 `delta.tool_calls`（先 name 后 arguments），
  再把 `finish_reason` 单独放一个 chunk（有工具调用时它必须排在 deltas 之后）。
- **容错**：`max_tokens` 截断导致块不完整时，解析失败就把原文照发，不静默丢弃、也不 500。

## 验证

```
非流式: finish_reason=tool_calls
        tool_calls=[{name:"Bash", arguments:{"command":"ls -la /tmp", ...}}]
        content=''  usage 正常
流式:   finish_reason=tool_calls
        call[0] name=Bash args={"command": "ls -la /tmp", ...}
        content 里无 DSML 泄漏
```

`tools/serve_test.py` 增加 7 项断言（非流式/流式各三项 + "DSML 未泄漏到 content"），
现在 **25 passed, 0 failed**。

## 附：`--always-usage` 已撤

上一轮为"网关不发 include_usage"加的 `--always-usage` 在确认 new-api 确实会发
`include_usage=1` 后已从运行参数移除（它会多一个 `choices: []` 的块，对不判空的客户端有风险）。
开关本身保留在代码里备用。
