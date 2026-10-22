# DeepSeek-V4.1-Flash-1017 在 H20-3e 下的 DeepInfer 推理服务

在 8×H20-3e 上用 [DeepInfer](https://github.com/DarkraiHL/DeepInfer) 的 SM90 分支跑 **DeepSeek-V4.1-Flash-1017**，
叠一层 OpenAI 兼容服务（文本 + 图片），并记录布局选型、性能实测和一处引擎侧修复
（解码吞吐 86.3 → 151.9 tok/s）。

本仓库只包含**我们自己的**产物：Dockerfile、工具脚本、补丁、调查笔记。
DeepInfer 源码不在此仓库内，用 `patches/deepinfer-h20-full.patch` 打到上游即可复现。

---

## 一、结论速查

| 问题 | 结论 |
|---|---|
| 8 卡上该用哪种并行布局？ | **CP8×EP8**（`--ep 8 --cp 8`，DP 自动为 1） |
| 1M 上下文能跑吗？ | 能。CP8 TTFT **20.403 s**；同一请求 DP8 要 **189.118 s** |
| 1M 是显存问题吗？ | 不是。实测 KV **891 B/token**，1M ≈ **0.87 GiB/rank** |
| 单会话解码差在哪？ | Engram fetch 的 padding/copy 冗余，已修复：CP8 **240.6–279.1 tok/s**，DP8 **151.9**（上游参考 158） |
| 高并发谁更好？ | c=32 时 DP8 吞吐 **2380.6** vs CP8 1777.1；但只要承诺 1M 上下文就只能用 CP8 |

**建议**：1M 上下文 + 并发 ≤32 的场景选 CP8。低于 ~8 并发 CP8 全面更优；
DP8 只在纯高吞吐（c≥32 短上下文）时领先，但它交付不了 1M。

---

## 二、硬件与软件基线

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA H20-3e，141 GB HBM3e/卡 |
| 驱动 | 595.91.07（`nvidia-driver-595-server`） |
| 内核 | 6.8.0-57-generic，Ubuntu 22.04，UEFI |
| 内存 | 2015 GB（可用 1953 GB） |
| 网络 | RoCE `mlx5_0` 100 Gb ACTIVE（`mlx5_1` DOWN）；`NCCL_IB_HCA=mlx5_0` |
| 镜像 | `lmsysorg/sglang:v0.5.22` + CUDA 13 + rdma-core 54，46.2 GB，单 Dockerfile |
| 模型 | `DeepSeek-V4.1-Flash-1017`，510.3 GB / 48 shards，fp4 routed experts，2 层 Engram |
| 权重占用 | 48.05 GiB/卡（EP8 分片 + 全量注意力） |
| 上下文 | `max_position_embeddings = 1048576`（模型原生） |

模型结构要点：40 层、`head_dim=512`、`num_key_value_heads=1`、稀疏注意力 `index_topk=512`、
`compress_ratios` 一半层做 2× 压缩；`engram_layer_ids=[1,14]`、`engram_head_dim=256`，
fp8 Engram 表分片后 23.4 GB/rank。

---

## 三、目录结构

```
docker/         单 Dockerfile（含 rdma-core 54 构建块）+ 构建脚本
patches/        deepinfer-h20-full.patch（相对上游 e205d42 的完整改动，19 文件 +2080）
                pr*.patch（分项补丁，便于阅读）
tools/          e2e 窗口、服务压测/回归、profiler、GIN/dma-buf 探针
notes/          01–20 调查记录（依赖基线、布局选型、网关排障、工具调用）
manifests/      源码仓库与子模块清单
```

---

## 四、快速开始

### 1. 构建镜像

```bash
cd docker && ./_run_build_final.sh        # 产出 deepinfer-sm90:v41-sm90-<ts> 与 :v41-sm90
```

`Dockerfile` 里有两条硬断言，构建失败即说明依赖没到位：

- `nm -D libmlx5.so.1.25.54.0 | grep mlx5dv_reg_dmabuf_mr` —— 该符号在 **rdma-core 54** 才引入（不是 51）；
- PR#99 落地检查（`NVLinkEngramBuffer` 与 `engram_nvlink.py` 存在）。

### 2. 启动服务（CP8）

```bash
serve/server.sh start --dp 8 --cp 8 --max-num-seqs 32 --port 8888
```

`--dp` 是 EP 大小（= 用几张卡），`--cp` 是上下文并行大小，引擎数 = `dp // cp`。
CP8 时只有一个引擎，所以 `--max-num-seqs` 直接决定整机并发上限。

必需的环境变量：

```bash
EP_DISABLE_GIN=1                 # 本机无 GIN；同时打开 PR#99 的 NVLink Engram 快路径（双关）
DEEPINFER_ENGRAM_NVLINK_FETCH=1
DEEPINFER_USE_NEXTN_DECODE=5     # MTP 深度
DEEPINFER_ENABLE_MTP_VARLEN=1    # MTP 真正的开关，默认 False；漏了会看到 accept=0%
DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact
DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140
```

### 3. 验证

```bash
python3 tools/serve_test.py        # 25 项 OpenAI 协议断言
```

---

## 五、性能实测

全部由仓库自带的 `examples/latency_benchmark.py` 跑出，同机、同 commit（`63b430a` + 本仓库补丁）、
同环境变量。TTFT 为 1 次预热 + 3 次测量取中位数（除非另注）；解码为 1024 in / 512 out。

### 5.1 首 token 延迟（TTFT）

| 布局 | 128K | 1M |
|---|---:|---:|
| **CP8 × EP8** | **3.064 s**（3.048 / 3.064 / 3.066） | **20.403 s** |
| DP8 × EP8 | 10.641 s（10.631 / 10.641 / 10.649） | 189.118 s |
| 上游参考 CP8（`doc/sm90.md`） | 2.114 s | 19.41 s |

1M 行是单次测量（`--ttft_warmup 0`）。记录里 `output_tokens=1, finish_reason=length`
是因为 `1M prompt + 32 输出 > max_model_len`，不是失败。

> 预填在 DP8 下只有一张卡算注意力，长度上去后超线性恶化：128K→1M 涨 17.8 倍，
> 而 CP8 只涨 6.7 倍。**这就是 1M 只有 CP8 可用的原因。**

### 5.2 解码吞吐与 TPOT

| 并发 | CP8 × EP8 | DP8 × EP8 | 上游参考 DP8 |
|---|---:|---:|---:|
| 1 | **240.6–279.1 tok/s**（TPOT 3.58–4.16 ms） | 151.9 tok/s（6.58 ms） | 158（6.33 ms） |
| 32 | 1777.1（15.25 ms，单请求 66.0） | **2380.6**（9.70 ms，单请求 103.4） | 2912（6.13 ms） |
| 64 | 未测 | **3826.7**（11.04 ms，单请求 91.0） | 4259（8.32 ms） |

c=32 时单请求 TTFT（1K 输入）：CP8 **2.30 s** / DP8 3.69 s。

### 5.3 Engram fetch 修复的收益

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| DP8 c=1 解码 | 86.3 tok/s（TPOT 11.5 ms） | **151.9**（6.58 ms） |
| DP8 c=32 吞吐 | 1615.7 | **2380.6** |
| DP8 c=64 吞吐 | 2770.6 | **3826.7** |
| CP8 c=1 解码 | 199.1–231.8 | **240.6–279.1** |
| CP8 c=32 吞吐 | 1720.9 | **1777.1** |

DP8 c=1 达到上游参考（158 tok/s）的 **96%**，TPOT 差 4%。

### 5.4 布局怎么选

| 并发 | 推荐 | 理由 |
|---|---|---|
| 1–8 | CP8 | 单会话与 TTFT 全面占优（1M 时 9.3×） |
| 9–32 | CP8（需 1M 时）或两者接近 | CP8 吞吐略低但 TTFT 好 34% |
| ≥32 纯短上下文 | DP8 | 聚合吞吐 +34%，但放弃 1M |

CP8 = 注意力 TP8 + 专家 EP8，8 卡服务同一 batch；DP8 = 8 个独立副本，一个请求只吃一张卡的注意力。
详见 [`notes/16-cp8-layout.md`](notes/16-cp8-layout.md)、[`notes/18-cp8-for-1m.md`](notes/18-cp8-for-1m.md)。

---

## 六、OpenAI 接口

**Base URL** `http://<host>:8888/v1` ｜ **模型名** `DeepSeek-V4.1-Flash-1017`（可用 `DEEPINFER_SERVED_MODEL_NAME` 改）

路由：`/v1/chat/completions`、`/v1/models`、`/health`、`/deepinfer/metrics`。
无鉴权（内网部署），`Authorization` 任意非空值即可。

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:8888/v1", api_key="x")

r = client.chat.completions.create(
    model="DeepSeek-V4.1-Flash-1017",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
)
print(r.choices[0].message.content, r.usage)
```

### 6.1 图片输入

```python
import base64
b64 = base64.b64encode(open("your.jpg", "rb").read()).decode()
client.chat.completions.create(
    model="DeepSeek-V4.1-Flash-1017",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "这张图里是什么？"},
    ]}],
    max_tokens=512,
)
```

- 支持 OpenAI `image_url` 与 Anthropic `image`+`source.base64` 两种写法，单请求可多图且保持顺序；
- 每图在 prompt 里落一个 `<｜deepseek_image｜>`，由引擎按 `compute_image_layout` 展开，
  `usage.prompt_tokens` **计入展开后的 token**（450×308 的图 = 189 token，与返回值逐位核对过）；
- 默认**只接受内联 base64**。该端点无鉴权，放开任意 URL 等于提供 SSRF 代理；
  需要 URL 时显式加 `--allow-image-urls`；
- 单图上限 64 MiB / `max_image_tokens=1024` 个 prompt token；解码失败返回 400，绝不静默降级为纯文本。

### 6.4 工具调用

`tools` 按 OpenAI 格式传入即可。请求侧由模型自带的编码器渲染进 prompt，响应侧由**同一个**
参考实现把模型写出的 DSML 块解析回 `tool_calls`——不解析的话，标记会原样出现在 `content` 里，
客户端（Claude Code 等）看到的是一段文本而不是工具调用。

```python
tools = [{"type": "function", "function": {
    "name": "Bash", "description": "Run a shell command",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]
r = client.chat.completions.create(
    model="DeepSeek-V4.1-Flash-1017",
    messages=[{"role": "user", "content": "列出 /tmp"}],
    tools=tools, max_tokens=300,
)
print(r.choices[0].finish_reason)          # tool_calls
print(r.choices[0].message.tool_calls)     # [{id, type, function:{name, arguments}}]
```

- 非流式：`finish_reason="tool_calls"`，`message.tool_calls[].function.arguments` 是 JSON 字符串；
- 流式：按 OpenAI 约定发 `delta.tool_calls`（先 `name` 后 `arguments`），`finish_reason` 单独一个 chunk；
- `max_tokens` 截断导致块不完整时，原文照发而不是丢弃，便于排查。

Anthropic 客户端（Claude Code）经 new-api 调用的链路已验证：我们返回 `tool_calls`，
new-api 转成 `content_block_start: tool_use` + `input_json_delta`。

### 6.3 经网关（new-api / one-api）使用

缓存命中数走标准字段 `usage.prompt_tokens_details.cached_tokens`，new-api 会记进用量日志的
`other.cache_tokens` 并显示为「缓存↓ N」。链路已验证，见
[`notes/19`](notes/19-cache-usage-through-gateway.md)。

若网关侧看不到缓存数，先看服务端日志而不是改协议——两个开关专为此准备：

```bash
--log-requests   # 每个请求打一行：[req] rid= stream= include_usage= prompt= cached=
--always-usage   # 流式一律下发 usage 块，供不发 include_usage 的老网关使用（默认关闭）
```

- 服务端 `cached=0` 且 `include_usage=1` → 真的没命中：前缀短于 128 token 的块，或服务刚重启（缓存是空的）。
- `include_usage=0` → 网关没要 usage，它只能自己数 token，缓存数必然为 0。new-api 由
  `FORCE_STREAM_OPTION`（新版本默认 true）控制；渠道开启「透传请求体」时也不会被注入。
  此时可让网关补上，或给服务加 `--always-usage`（代价：流式多一个 `choices: []` 的块，
  对不判空直接取 `choices[0]` 的客户端有风险）。

### 6.2 参数支持

**生效**：`max_tokens`、`max_completion_tokens`、`temperature`、`top_p`、`seed`、`stream`、
`stream_options.include_usage`、`tools`（双向：渲染进 prompt，并把模型输出的 DSML 解析成
`tool_calls`）、`prefix_cache`（非标准，默认 true）。

**未实现（会被忽略）**：`stop`、`stop_token_ids`、`top_k`、`min_tokens`、`n>1`、`logit_bias`、
`presence_penalty`/`frequency_penalty`、`response_format`、`reasoning_effort`。

`usage.prompt_tokens_details.cached_tokens` 与 vLLM 口径一致；前缀缓存命中时
`/deepinfer/metrics` 另有 `hicache` 计数。

---

## 七、关键修复：Engram fetch 的隐藏开销

**症状**：DP8 单会话解码只有 86 tok/s，上游参考 158。

**定位**：torch profiler 把 58.9 ms 的解码步拆开，Engram 占 41.9 ms，其中
`masked_fill` 13.30 + `copy_` 11.14 = **24.4 ms 是纯冗余**，真正的跨卡传输只 8.72 ms。
给 `fetch` 加探针后看到根因：

```
num_tokens=32768 max=32768   ← 预填
num_tokens=24    max=32768   ← 解码每一步
```

解码一步只有 24 个 token，却按预填容量 32768 走集合通信，**padding 放大 1365 倍**，
`_send`/`_recv` 各 3 GB（合计 6 GB 显存）。

**改法**（`deepinfer/model_executor/parallel_utils/engram_nvlink.py`）：

1. `index_select(..., out=)` 直接把行写进发送缓冲，省掉一次 3 GB 拷贝；
2. 删掉 `masked_fill_` —— 接收侧只读拥有该行的 rank 那一片，其余槽位是死字节，**从不被读**；
3. 缓冲改 layer-major，按层做 `all_to_all`。

**正确性**由 `tests/distributed/test_engram_nvlink.py` 保证：以"全表 all_gather + 朴素索引"为参考实现，
覆盖本地/远程归属、两层不同表大小、**各 rank token 数不同**三种情形，逐字节比对通过。

**结果**：DP8 单会话 86.3 → 151.9 tok/s（上游参考的 96%），c=32 1615.7 → 2380.6。

一个踩过的坑：不能按"预填/解码"分阶段切换缓冲大小。DP8 各副本独立，一个副本在预填（cap=32768）
而另一个在解码（cap=24）时 `all_gather` 形状不一致，NCCL 会永久死锁。最终用与负载无关的常量 cap。

---

## 八、测试与验证

| 项目 | 结果 |
|---|---|
| OpenAI 协议回归（`tools/serve_test.py`） | **25 passed, 0 failed** |
| 非流式 / 流式 SSE / usage / cached_tokens | 全部通过 |
| 前缀缓存 | 同 prompt 二次请求 `cached_tokens=3840/4019` |
| 图片输入 | 单图、双图、流式、Anthropic 块、URL 拒绝，全部通过 |
| 工具调用 | 非流式/流式均返回 `tool_calls`，无 DSML 泄漏 |
| Engram fetch 正确性（2 卡） | 3 组情形逐字节一致 |
| 图片 token 数学 | 410 个尺寸与引擎 `compute_image_layout` 比对，**0 处不一致** |

---

## 九、已知限制

- **GIN 不可用**：本机 `nvidia-peermem.ko` 是 7.9 KB 空壳，
  `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0`（改 `iommu=pt` 也救不回来），
  DeepEP 只能走 NCCL 回退。这大概率是 CP8 1M 比参考慢 5%、c=32 吞吐低于参考的原因，未做最终证明。
  详见 [`notes/06`](notes/06-engram-gin-rootcause.md)–[`notes/08`](notes/08-nvidia-escalation.md)。
- **CP8 的 c=64 未测**（需求只到 32）。
- **多轮图文**：历史轮次的图片会被重新送、重新计费 prefill；图像段的前缀缓存命中未验证。
- `BATCH_INVARIANT` 在 SM90 不支持（上游限制）。
- 服务无鉴权，仅适合内网；对外暴露前需自行加网关。

---

## 十、复现补丁

```bash
git clone https://github.com/DarkraiHL/DeepInfer.git
cd DeepInfer && git checkout e205d42e4b28185e50ce30f6b7204cd01e59974c
git apply /path/to/patches/deepinfer-h20-full.patch
```

补丁内容（19 文件，+2080/−1）包含：SM90 支持、前缀缓存（PR#144）、scheduler API（PR#195）、
OpenAI 服务层（PR#147）、NVLink Engram fetch（PR#99），以及本仓库的 Engram 优化、
`--cp` 透传、图片输入。分项补丁见 `patches/pr*.patch`。

---

## 十一、参考

- DeepInfer: <https://github.com/DarkraiHL/DeepInfer>
- 上游方法论：`doc/sm90.md`、`doc/inference_deployment_qa.md`
- 本仓库调查记录：`notes/01`–`notes/20`，其中
  [`notes/17`](notes/17-engram-fetch-fix.md)（Engram 修复）与
  [`notes/18`](notes/18-cp8-for-1m.md)（1M 布局选型）、
  [`notes/19`](notes/19-cache-usage-through-gateway.md)（网关下的缓存显示排查）、
  [`notes/20`](notes/20-tool-calls-dsml.md)（工具调用解析）是核心。
