# 22 — 消息顺序到底乱没乱：逐层验证

现象：经 Claude Code → new-api → 本服务时，怀疑消息前后顺序错乱。

## 结论

本服务侧**不重排**，缓存**不可能**给出别的会话的历史。两处都有硬证据，不是推断。

## 1) 缓存：内容寻址，跨会话串味不可能

`deepinfer/hicache/host_cache.py:8` 原文：

> Pages are indexed by a hash chain over the prompt tokens
> (page i's key covers tokens [0, (i+1)*block_size))

实现（:178-182）：

```python
prev = hashes[-1] if hashes else b""
for i in range(len(hashes), num_pages):
    prev = hashlib.blake2b(prev + toks[i*bs:(i+1)*bs].tobytes(), digest_size=16).digest()
```

每一页的 key 是**从第 0 个 token 起累积哈希**。前缀只要有一个 token 不同，后续所有页的 key 都不同。
所以命中 ⇒ 前缀完全一致；**不存在"命中了一段属于别的对话的 KV"这种可能**。

## 2) 编码器：保序，实测

用 `encoding.encode_messages` 直接跑三种真实形态（单工具、多工具结果逆序、多轮连续工具），
把 prompt 的说话人标签抽出来：

| 输入 roles | 输出标签 |
|---|---|
| system, user, assistant(tool_calls), tool, user | `U A U A` |
| system, user, assistant(2 calls), tool(B), tool(A), user | `U A U A` |
| system, user, asst, tool, asst, tool, user | `U A U A U A` |

`U/A` 严格按输入顺序展开，唯一的合并是 `merge_tool_messages` 把 `<tool_result>` 和其后的用户文本
放进同一个 user 轮（DeepSeek 格式没有独立 tool role，这是参考实现的既定行为）。

线上还看 `encode_fallbacks_total`：**0**，说明粗糙的降级拼接路径从未被触发。

## 3) new-api：代码 + 自带 golden 测试都保序

`relaykit/relayconvert/internal/claude_messages/to_oai_chat_req.go`：

- system 一律插在 index 0；
- 每条 Claude 消息内先收集 `tool_use`→`tool_calls`、`tool_result`→独立的 `tool` 消息，
  再把该消息本身 append，**不排序、不反转**；
- `unnamedToolResults` 只是按记录的 index 回填 tool 名字，不动消息位置。

它自己的测试也印证：输入 `[tool_result(toolu_1), text]` → `Messages[2]=tool, Messages[3]=user`；
两个 tool_result 时 `Messages[2].ToolCallId == "toolu_shot"`（第一个对应第一个）。

唯一会被 OpenAI 格式强制改变的是**同一条 Claude 消息内** text 与 tool_result 的相对位置：
tool 消息必须紧跟在 assistant 之后，所以即使 text 在前，输出里也是 tool 在前。
这是协议约束，不是乱序。

## 4) 新增的取证日志

`--log-requests` 下每个请求打两行：

```
[msgs] in=system:20 | user:14 | assistant:0(call_1) | tool:11->call_1 | user:10
[msgs] out=UAUA
```

- `in=`：从网关收到的角色顺序、内容长度、tool_call id、tool→call 的对应关系；
- `out=`：这些内容在最终 prompt 里的说话人顺序。

以后怀疑乱序，直接对这两行：`in=` 与客户端发出的顺序不一致 → 网关问题；
`in=` 一致但 `out=` 与预期不符 → 本服务问题。

实测一次完整调用（含工具）输出：finish_reason=tool_calls，
tool_calls=[Bash: `rm /tmp/a.txt && ls /tmp`]，行为与合并后的上下文一致。
