# DeepSeek-V4.1-Flash-1017 在 H20-3e 下的 DeepInfer 推理服务

在 8×H20-3e 上用 [DeepInfer](https://github.com/DarkraiHL/DeepInfer) 的 SM90 分支跑 **DeepSeek-V4.1-Flash-1017**，
对外提供三套会话协议（Chat Completions / Responses / Anthropic Messages，全部支持流式、图片输入、
思考模式和缓存命中上报），协议转换交给 DeepSeek 官方的
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)，并记录布局选型、性能实测和两处引擎侧修复
（Engram 取表的冗余拷贝、集合通信缓冲按容量复用）。

本仓库只包含**我们自己的**产物：Dockerfile、工具脚本、补丁、调查笔记。
DeepInfer 源码不在此仓库内，用 `patches/deepinfer-h20-full.patch` 打到上游即可复现。

---

## 一、结论速查

| 问题 | 结论 |
|---|---|
| 8 卡上该用哪种并行布局？ | **CP8×EP8**（`--ep 8 --cp 8`，DP 自动为 1） |
| 1M 上下文能跑吗？ | 能。CP8 TTFT **30.2 s**；同一请求 DP8 要 **141.0 s**（§5.1） |
| 1M 是显存问题吗？ | 不是。实测 KV **891 B/token**，1M ≈ **0.87 GiB/rank** |
| 单会话解码多少？ | CP8 中位 **276.5 tok/s**（5 次：134–375，跨度来自 MTP 接受率），DP8 223.5（§5.2） |
| 高并发谁更好？ | c=32 DP8 **2605.7** vs CP8 1614.3（+61%）；但只要承诺 1M 就只能 CP8 |
| GIN 要不要开？ | 生产开着。关掉（NVLink 回退）1M 预填快 16%，但专家通信一起退回 NCCL（§5.3） |
| 消息顺序会不会乱？ | 不会。渲染/解析/流式都走官方 deepseek-recipe，与 checkpoint 自带编码器逐 token 对表 10/10（[`notes/24`](notes/24-message-order-vs-official-encoding.md)、[`notes/25`](notes/25-recipe-protocol-layer.md)） |
| 协议转换谁来做？ | **deepseek-recipe**（官方 Rust 实现，已在镜像里）。服务只保留"token ids 进、token ids 出"，顺带拿到 `/v1/messages`、`/v1/responses`、`reasoning_content`、缓存命中字段（[`notes/25`](notes/25-recipe-protocol-layer.md)） |

**建议**：承诺 1M 上下文、或并发 ≤16 时选 CP8；DP8 只在纯短上下文高吞吐（c≥32）时值得，
它的 c=32 聚合吞吐高 61%，但 1M 预填慢 4.7 倍。

---

## 二、硬件与软件基线

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA H20-3e，141 GB HBM3e/卡 |
| 驱动 | 595.91.07 **开源内核模块**（`nvidia-driver-595-server-open`），GIN 可用 |
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
patches/        deepinfer-h20-full.patch（相对上游 e205d42 的完整改动）
                pr*.patch（分项补丁，便于阅读）
tools/          服务层回归/压测、协议对表、profiler、GIN/dma-buf 探针（索引见 tools/README.md）
notes/          01–25 调查记录（索引见 notes/README.md）
data/           冻结的测量结果：基准 harness 的 summary、协议/缓存测量
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

> 部署形态：容器里 `/opt/deepinfer/DeepInfer` 是完整 checkout（构建产物在里面），
> 但 `serve/*.py` 是**逐文件**从本仓库挂进去的。改了文件名或新增文件，要重建容器或
> `docker cp` 覆盖，否则容器里跑的还是旧文件——这个坑在本轮改名时踩到过。

必需的环境变量：

```bash
DEEPINFER_USE_NEXTN_DECODE=5     # MTP 深度
DEEPINFER_ENABLE_MTP_VARLEN=1    # MTP 真正的开关，默认 False；漏了会看到 accept=0%
DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact
DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140
```

GIN 需要**开源内核模块**（见[九、已知限制](#九已知限制)）。换好之后不要再设
`EP_DISABLE_GIN=1`——那会让 DeepEP 回退到 NCCL 并启用 NVLink Engram 回退路径，
实测比 GIN 慢约 14%（TPOT 6.5 vs 5.6 ms）。

### 3. 验证

```bash
python3 tools/serve_test.py        # 26 项协议断言
```

更多工具（压测、缓存矩阵、协议对表、GIN 探针）见 [`tools/README.md`](tools/README.md)。

---

## 五、性能实测

全部由仓库自带的 `examples/latency_benchmark.py` 跑出。测量条件、命令和每条请求的明细都在
[`data/README.md`](data/README.md)，下面的表由 `python3 tools/summarize_data.py data/r2`
从冻结的 `summary.json` 生成，不是手抄的。

- **CP8** = `--ep 8 --cp 8`：注意力 TP8 + 专家 EP8，8 张卡服务同一个 batch（DP 自动为 1）。
- **DP8** = `--ep 8 --cp 1 --dp 8`：8 个独立副本，一个请求只吃一张卡的注意力。

### 5.1 首 token 延迟（TTFT）

| 布局 | 128K | 1M |
|---|---:|---:|
| **CP8 × EP8** | **3.859 s**（3.843 / 3.859 / 3.914） | **30.220 s** |
| DP8 × EP8 | 14.297 s（14.249 / 14.297 / 14.481） | 140.970 s |

1M 行是单次测量（`--ttft_warmup 0`）：光预填就要 30 s，预热一次太贵。

输入从 128K 到 1M 涨 7.9 倍，CP8 的 TTFT 涨 7.8 倍（近线性），DP8 涨 9.9 倍——
DP8 的预填只有一张卡算注意力，长度上去后超线性恶化。**承诺 1M 上下文时只有 CP8 可用（快 4.7 倍）。**

### 5.2 解码吞吐与 TPOT（4096 in / 1024 out，`ignore_eos` 生成满长度）

| 并发 | CP8 总量 | CP8 单请求 | CP8 TPOT | DP8 总量 | DP8 单请求 | DP8 TPOT |
|---|---:|---:|---:|---:|---:|---:|
| 1 | — | **276.5**（5 次） | 3.62 ms | — | 223.5（3 次） | 4.47 ms |
| 32 | 1614.3 | 70.5 | 12.59 ms | **2605.7** | 148.8 | **6.53 ms** |
| 64 | 1836.4 | 41.3 | 21.74 ms | **3693.9** | 109.8 | **8.70 ms** |

单会话那一行不要用一次测量代表：同一配置连测 5 次是 **134 / 251 / 277 / 366 / 375 tok/s**
（原始记录 `data/r2/single-cp8-gin/`）。3 倍跨度来自 MTP draft 的接受长度——accept 1.80 时
TPOT 7.47 ms、accept 5.72 时 2.67 ms，是**同一份权重的真实分布**，不是抖动。表里取这 5 次的中位数；
同一配置另一次 3 次测量（`data/r2/dec-cp8/`）中位是 243.8，差异同样来自接受率。

并发 32/64 时的 TTFT（4K 输入）：CP8 2.22 s / 3.71 s，DP8 3.07 s / 4.94 s。

### 5.3 GIN 开还是关

换开源内核模块后 GIN 可用，DeepEP 走原生 GIN；显式关掉
（`EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1`）则回退到 NCCL 集合通信 +
PR#99 的 NVLink Engram 取数。同机、同驱动、同一天，只切这两个变量：

| 指标 | GIN（生产默认） | 关闭 GIN（NVLink 回退） |
|---|---:|---:|
| TTFT @128K | 3.859 s | **3.342 s**（快 13%） |
| TTFT @1M | 30.220 s | **25.274 s**（快 16%） |
| c=1 解码，5 次中位 | **276.5** | 221.5 |
| c=32 总吞吐 | 1614.3 | 1665.8 |
| c=32 TPOT 中位 | 12.59 ms | 12.45 ms |

- **长上下文预填，回退路径更快**，而且离散度很小（3.328–3.373 s，三次重复）。差在 Engram
  取表：预填一步 32K token 的取表走 GIN 是细粒度 RDMA，批量上可能不占优；回退路径用集合通信
  一次搬完。机制未逐层 profiling（方法见 [`notes/15`](notes/15-profiler-engram-breakdown.md)），
  但方向在两种输入长度上一致。
- **短并发解码两者打平**：c=32 总吞吐差 3%，TPOT 差 0.14 ms。
- **单会话那一列两组分布重叠**（GIN 134–375，NVLink 111–312），差异不足以归因到传输层——
  真正的变量是每次请求的 MTP 接受率。
- 关 GIN 会让 DeepEP 的专家通信一起退回 NCCL，这是"没有 GIN 时的可用路径"，不是免费开关。
  **要极致 1M 预填可以切过去，但必须在自己负载上复测。**

生产跑的是 GIN（驱动修好后的默认路径，见 [`notes/23`](notes/23-driver-swap-gin.md)）。

### 5.4 布局怎么选

| 场景 | 推荐 | 理由 |
|---|---|---|
| 并发 ≤16，或需要 1M | **CP8** | 单会话更快，1M 预填快 4.7 倍 |
| 并发 ≥32，纯短上下文 | **DP8** | c=32 +61%、c=64 +101% 聚合吞吐 |
| 既要 1M 又要高吞吐 | CP8 + 上游限流 | 1M 请求会把 8 张卡绑成同一个 batch |

布局的依据是 [notes/16](notes/16-cp8-layout.md) 与 [notes/18](notes/18-cp8-for-1m.md)；
注意那两篇是 10-22 的数字（专有驱动 + 关 GIN），与本节口径不同，只作推理过程的参考。

---

## 六、对外接口（三协议）

**Base URL** `http://<host>:8888/v1` ｜ **模型名** `DeepSeek-V4.1-Flash-1017`（可用 `DEEPINFER_SERVED_MODEL_NAME` 改）

路由：`/v1/chat/completions`、`/v1/messages`（Anthropic，另有 `/anthropic/v1/messages` 这个官网同款前缀）、
`/v1/messages/count_tokens`、`/v1/responses`、`/v1/models`、`/health`、`/deepinfer/metrics`。
三个协议端点共用同一个引擎和同一套转换/解析实现（见 §6.5）。
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

- 形状跟着端点走：`/v1/chat/completions` 收 OpenAI `image_url`，`/v1/messages` 收 Anthropic
  `image` + `source.base64`（把 Anthropic 形状发到 chat 端点会 400，OpenAI schema 就是这么定的）；
- 单请求可多图且保持顺序，图片也可以落在 `tool_result` 里；
- 每图在 prompt 里落一个 `<｜deepseek_image｜>`，由引擎按 `compute_image_layout` 展开，
  `usage.prompt_tokens` **计入展开后的 token**（450×308 的图 = 189 token，与返回值逐位核对过）；
- 默认**只接受内联 base64**。该端点无鉴权，放开任意 URL 等于提供 SSRF 代理；
  需要 URL 时显式加 `--allow-image-urls`；
- 单图上限 64 MiB / `max_image_tokens=1024` 个 prompt token；解码失败返回 400，绝不静默降级为纯文本。

### 6.2 工具调用

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
- `max_tokens` 截断导致块不完整时，原文照发而不是丢弃，便于排查；
- 块的识别按官方解析器的口径（`NewlinesAndMarker`：**任意个换行、含 0 个**，再加 `<｜DSML｜`），
  标签之间多一个空格也认——比"只认 `\n\n` 开头"更鲁棒，见
  [`notes/24`](notes/24-message-order-vs-official-encoding.md)。

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

### 6.4 参数支持

**生效**：`max_tokens`、`max_completion_tokens`、`temperature`、`top_p`、`top_k`、`seed`、`stop`、
`stream`、`stream_options.include_usage`、`tools`（双向：渲染进 prompt，并把模型输出的 DSML 解析成
`tool_calls`）、`reasoning_effort` / `thinking`（选思考模式与努力档，输出切到
`reasoning_content`）、`response_format`（json_schema / regex，校验并写进 prompt）、`prefix_cache`
（非标准，默认 true）。

**会被接受但不生效**：`presence_penalty` / `frequency_penalty`（DeepSeek 官方 API 也不支持，校验后丢弃）、
`logprobs`、`min_tokens`、`logit_bias`、`stop_token_ids`。

**会被拒绝**：`n > 1`（400）、`max_tokens` 超出上下文剩余（400）。

`usage` 里带缓存命中：chat completions 是 `prompt_tokens_details.cached_tokens` +
`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，messages 是 `cache_read_input_tokens`；
思考模式还有 `completion_tokens_details.reasoning_tokens`。前缀缓存命中时
`/deepinfer/metrics` 另有 `hicache` 计数。

### 6.5 协议层：deepseek-recipe

请求转换、prompt 渲染/编码、输出解析都交给 [`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe)
（DeepSeek 官方的 Rust 实现，已装在镜像里）。服务只负责"给 token ids、吐 token ids"这一段。

```bash
serve/server.sh start --dp 8 --cp 8 --max-num-seqs 32 --port 8888                        # 默认 recipe
serve/server.sh start --dp 8 --cp 8 --max-num-seqs 32 --port 8888 --protocol-layer builtin  # 旧路径，A/B
```

和 checkpoint 自带参考编码器逐 token 对表：10/10 用例 prompt 与 token ids 相同；
控制符注入（用户文本里写 `<｜Assistant｜>`）在旧路径会变成真控制符，新路径只是文字。
风险与取舍见 [`notes/25`](notes/25-recipe-protocol-layer.md)。

---

### 6.6 和 DeepSeek 官网接口的对照

官网（[api-docs.deepseek.com](https://api-docs.deepseek.com/)）当前是四套协议 + 四个配套端点。
本服务逐项实测如下（✅ 已实测通过，⚠️ 接受但不完全生效，❌ 未实现）：

| 官网能力 | 本服务 | 实测 |
|---|---|---|
| Chat Completions `POST /chat/completions` | ✅ `/v1/chat/completions` | 流式/非流式、tools、usage |
| Responses API `POST /responses` | ✅ `/v1/responses` | `output[].content[].output_text` |
| Anthropic `base_url=…/anthropic` | ✅ `/anthropic/v1/messages`（`/v1/messages` 同样可用） | 事件流、`cache_read_input_tokens`；`claude-*` 模型名自动映射（同官网） |
| `POST /messages/count_tokens` | ✅ `/v1/messages/count_tokens` | 15 == `usage.input_tokens` 15 |
| 思考模式 `thinking` / `reasoning_effort` | ✅ | `reasoning_content`、`reasoning_tokens` |
| JSON Output `response_format` | ✅ `json_object` / `json_schema` | 写进 prompt 并约束格式 |
| Tool Calls + `tool_choice` | ✅ 含 `required`/指定函数 | 非思考模式下 |
| Chat Prefix Completion（官网在 `/beta`） | ✅ 主端点直接支持 `prefix: true` | 只回续写部分，不回显前缀 |
| 视觉输入 | ✅ 图片块 | 768×768 → prompt_tokens 393 |
| Context Caching | ✅ 前缀缓存 | `cached_tokens` / `prompt_cache_hit_tokens` |
| FIM Completion（`/beta/completions`） | ❌ | checkpoint 有 `<｜fim▁begin｜>` 等 token，但两份参考实现都没有 FIM 模板 |
| Files API | ❌ | 未实现 |
| Search & Fetch（服务端联网） | ❌ | 需要搜索后端 |
| `GET /user/balance`、API Key 鉴权、计费 | ❌ | 内网无鉴权，无计费概念 |
| `logprobs` / `top_logprobs` | ⚠️ 接受但引擎不产出 | `SamplingParams.logprobs` 在引擎侧是兼容字段 |
| `n > 1` | ❌ 400 | OpenAI 契约与引擎都只做单条 |

即：**会话类协议四套里三套齐了**（Chat Completions / Responses / Anthropic），
外加官网要 `/beta` 才有、这里直接在主管线上的 prefix completion 和 JSON Output；
缺的是 FIM、Files、联网、鉴权和计费——前三个都能做，但都不是"接到模型上"这件事的一部分。

Claude Code 直接按官网的写法接：

```bash
export ANTHROPIC_BASE_URL=http://<host>:8888/anthropic
export ANTHROPIC_API_KEY=x      # 无鉴权，随意非空
```

### 6.7 缓存命中怎么上报

三个协议都上报，字段按各自规范，同一段前缀在三处命中数相同（实测都是 2560）：

| 协议 | 字段 | 非流式 | 流式 |
|---|---|---|---|
| Chat Completions | `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`（官网同名字段）+ `prompt_tokens_details.cached_tokens`（OpenAI 字段） | ✅ | ✅ 最后一个 chunk |
| Responses | `usage.input_tokens_details.cached_tokens`（官网 Responses 用的就是这个字段） | ✅ | ✅ `response.completed` / `response.incomplete` |
| Messages | `usage.cache_read_input_tokens` + `cache_creation_input_tokens` | ✅ | ✅ `message_start` 与 `message_delta` |

实测（同一段 2818 token 的 prompt，第二次请求）：

```
chat       prompt_tokens=2818  prompt_cache_hit_tokens=2560  prompt_cache_miss_tokens=258
           prompt_tokens_details.cached_tokens=2560
responses  input_tokens=2818   input_tokens_details.cached_tokens=2560
messages   input_tokens=258    cache_read_input_tokens=2560    cache_creation_input_tokens=0
```

Messages 的 `input_tokens` **不含**命中部分，这是 Anthropic 自己的语义：
`input_tokens + cache_read_input_tokens` 才是总输入（258 + 2560 = 2818）。

三处共享同一份缓存——同一段 prompt 先用 chat 端点打过，再发到 messages 端点，第一次就命中 2560。

两处与官网的差异，都是设计选择：

- `cache_creation_input_tokens` 恒为 0：缓存是自动的。官网在 Anthropic 面同样把 `cache_control` 标为忽略。
- 粒度不同：官网按"前缀单元"（请求边界 / 公共前缀 / 固定间隔）切分持久化；
  我们按 128-token 块链做内容寻址，并保留 `keep_tokens=140` 尾部重算。上报口径一致，命中位置会有差异。
- 流式响应即使没带 `stream_options.include_usage` 也会带上 usage：网关（new-api/one-api）只能从上游
  usage 拿到计费与缓存数字，丢掉它会退化成网关自己分词、缓存永远算 0。

工具：[`tools/cache_hit_matrix.py`](tools/cache_hit_matrix.py) 一条命令复跑上面这张表。

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
3. 缓冲改成 layer-major（`send[layer]` 连续，`out=` 才写得快），并按容量缓存复用——
   一次引擎只用到两个容量（解码与预填），没必要每次 fetch 重开三块大缓冲。

**正确性**由 `tests/distributed/test_engram_nvlink.py` 保证：两卡 `torchrun` 跑，以"全表 all_gather +
朴素索引"为参考实现，覆盖本地/远程归属、两层不同表大小、**各 rank token 数不同**、远低于容量的
小步、以及 scaling-factor 的 gather，逐字节比对：

```bash
torchrun --nproc_per_node=2 tests/distributed/test_engram_nvlink.py
[test] PASS (0 mismatching layer-cases)
```

**结果**：DP8 单会话 86.3 → 151.9 tok/s（上游参考的 96%），c=32 1615.7 → 2380.6。

**请注意适用范围**：这条路径只在**关掉 GIN**时走（`EP_DISABLE_GIN=1` 会自动打开它），也就是没有
GIN 时的回退路径。换开源驱动后生产走 GIN + DeepEP 原生 Engram，不经过这里；但 §5.3 的对照说明
回退路径仍有价值——它的长上下文预填更快，探针和这个修复都得留着。

一个踩过的坑：不能按"预填/解码"分阶段切换缓冲大小。DP8 各副本独立，一个副本在预填（cap=32768）
而另一个在解码（cap=24）时 `all_gather` 形状不一致，NCCL 会永久死锁。最终用与负载无关的常量 cap。

---

## 八、测试与验证

最近一轮（2026-10-23，`serve/deepseek_server.py`）：原始输出都在
[`data/r2/protocol/`](data/r2/protocol/)。

| 项目 | 结果 |
|---|---|
| 协议回归（`tools/serve_test.py`） | **26 passed, 0 failed** |
| 镜像自检（`docker/verify_service.py`） | **7 项 PASS**（含 serve 层可导入、recipe 0.2.0、app factory） |
| 协议层对表（`tools/recipe_parity.py`） | prompt 与 token ids **10/10 相同**；输出解析 5/5；控制符注入、图片占位符通过 |
| 缓存命中矩阵（`tools/cache_hit_matrix.py`） | 同一段 2818 token 前缀，三协议都命中 **2560**，字段各按自己的规范 |
| 图片输入 | 单图、Anthropic 块走 `/v1/messages`、形状不符 400、URL 拒绝，全部通过 |
| 工具调用 | 非流式/流式均返回 `tool_calls`，无 DSML 泄漏 |
| Engram fetch 正确性（2 卡 torchrun） | **0 mismatching layer-cases** |
| 图片 token 数学 | 410 个尺寸与引擎 `compute_image_layout` 比对，**0 处不一致** |

---

## 九、已知限制

- **GIN 依赖开源内核模块**：专有模块下 `nvidia-peermem` 是桩（`init_module` 恒返回 `-EINVAL`），
  且 `cuDeviceGetAttribute(124, DMA_BUF_SUPPORTED)=0`，NCCL 因此拒绝 GIN。
  换成 `nvidia-driver-595-server-open`（同版本，只换内核模块）后 attr 124 变 1、GIN 探针通过。
  过程与回滚见 [`notes/23`](notes/23-driver-swap-gin.md)，可行性分析见 [`notes/21`](notes/21-gin-fixability.md)。
- **单会话解码的跨度是固有的**：同一配置 5 次测量 134–375 tok/s，变量是 MTP draft 的接受率
  （accept 1.80 → 5.72），不是服务抖动；对外报数请用中位数 + 区间。
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

补丁内容（**19 文件，+2601 / −1**）包含：SM90 支持（#92）、前缀缓存（#144）、scheduler 绑定（#195）、
服务层（#147，`serve/deepseek_server.py`）、NVLink Engram fetch（#99），以及本仓库的 Engram 优化、
`--cp` 透传、图片输入、模型名可配。

已在干净的上游基线上 `git apply --check` 并真实应用过一次（应用后的工作树与本仓库记录一致）。
`patches/history/` 里是按 PR 拆开的中途快照，**不要单独应用**（[说明](patches/README.md)）。

---

## 十一、参考

- DeepInfer: <https://github.com/DarkraiHL/DeepInfer>
- 上游方法论：`doc/sm90.md`、`doc/inference_deployment_qa.md`
- 本仓库调查记录：`notes/01`–`notes/25`（[索引](notes/README.md)），其中
  [`notes/17`](notes/17-engram-fetch-fix.md)（Engram 修复）与
  [`notes/18`](notes/18-cp8-for-1m.md)（1M 布局选型）、
  [`notes/19`](notes/19-cache-usage-through-gateway.md)（网关下的缓存显示排查）、
  [`notes/20`](notes/20-tool-calls-dsml.md)（工具调用解析）、
  [`notes/23`](notes/23-driver-swap-gin.md)（换开源模块修好 GIN）、
  [`notes/24`](notes/24-message-order-vs-official-encoding.md)（消息顺序与官方实现对表）是核心。
- DeepSeek 官方协议框架：<https://github.com/deepseek-ai/deepseek-recipe>（Claude/OpenAI/Responses
  三协议互转 + 流式解码；`notes/24` 用它作为 prompt 渲染与输出解析的对照实现）
