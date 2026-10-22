# 19 — 经 new-api 网关时"缓存↓"不显示的排查

现象：new-api 的用量日志里看不到 `缓存↓ N`。2026-10-22 复现时，new-api 记录
`31,565 / 159, 缓存↓ 31,104`，服务端同一请求的日志为
`[req] rid=0ff79fea stream=1 include_usage=1 images=0 prompt=31565 cached=31104`——
链路其实是通的。

## 链路

```
serve：usage.prompt_tokens_details.cached_tokens
  → new-api relaykit/dto：PromptTokensDetails.CachedTokens
    （billing_usage.go:276 另有兜底：CachedTokens==0 时取 prompt_cache_hit_tokens）
  → service/text_quota.go:256  summary.CacheTokens = usage.PromptTokensDetails.CachedTokens
  → service/log_info_generate.go:106  other["cache_tokens"]
  → 前端 usage-logs 读 other.cache_tokens 渲染"缓存↓"
```

## 两个真实成因

1. **请求确实没命中**。host 前缀缓存按 128 token 块对齐，短于一个块的前缀永远不命中；
   服务重启会清空缓存，重启后的第一批请求必然 `cached=0`。
   实测：同一段 5000 token 文档配不同问题 → 首问 0，第二问 4864。
2. **网关没发 `stream_options.include_usage`**。OpenAI 规范里流式默认不回 usage，
   我们按规范实现；若网关不发这个字段，就拿不到 usage，只能自己数 token，缓存数自然为 0。
   new-api 的开关是 `FORCE_STREAM_OPTION`（`common/init.go:193`，新版本默认 true）。
   另外若渠道开了"透传请求体"，也不会被注入。

## 为此新增的两个运维开关

- `--log-requests`：每个请求结束时打一行
  `[req] rid= stream= include_usage= images= prompt= cached= aborted=`。
  一眼区分"网关没要 usage"和"确实没命中"。
- `--always-usage`：流式响应一律下发 usage 块，即使客户端没要。
  为不发 `include_usage` 的老网关准备；**注意**这会多一个 `choices: []` 的块，
  对不判空直接取 `choices[0]` 的粗糙客户端有风险，所以默认关闭。

## 结论

字段名、格式、记账都没有问题，不需要改协议。排查顺序建议：
先看服务端 `[req]` 日志的 `cached=`，再看 `include_usage=`，最后才怀疑网关。
