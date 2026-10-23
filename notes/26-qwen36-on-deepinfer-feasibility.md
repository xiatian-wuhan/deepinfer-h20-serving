# Qwen3.6-35B-A3B 迁到 DeepInfer：可行性评估与改造量

调研日期 2026-10-23。目标模型事实取自 HF 仓库 `Qwen/Qwen3.6-35B-A3B`（config.json / README.md / tree API），
DeepInfer 事实取自 `deepinfer-sm90-92-p195-p147` @ `ba9203a`（线上跑的那一份）。

## 0. 结论

| 问题 | 答案 |
|---|---|
| 技术上行不行 | 行，但性质不是「适配」，是**在 DeepInfer 里新写一个模型栈** |
| 要多大改造 | 最小可用（纯文本/无 MTP/单卡）**80–130 人日**；生产级（含 MTP、多模态、CP、调优）**120–190 人日** |
| 建议 | **不做**。同一台 H20 上 vLLM / SGLang 已原生支持，1–2 天可上线 |

判断依据只有一条：**DeepInfer 的四个核心资产 —— MLA、FP8/FP4 专家、Engram、CP 长上下文 —— 在这个模型上一个都用不上**，
而它缺的恰好是目标模型的三根支柱：GQA paged attention、Gated DeltaNet、softmax 门控 MoE。

## 1. 目标模型到底长什么样

`architectures = Qwen3_5MoeForConditionalGeneration`，`model_type = qwen3_5_moe`（text 部分 `qwen3_5_moe_text`）。

| 项 | 值 | 对引擎的含义 |
|---|---|---|
| 层布局 | 40 层 = 10 × ( 3×(GatedDeltaNet→MoE) + 1×(GatedAttention→MoE) ) | **混合注意力**，每层类型不同 |
| Gated Attention（10 层） | 16 Q / 2 KV，head_dim 256，RoPE 只转 64 维，`attn_output_gate=true` | MQA-latent 之外的 GQA + 部分 RoPE + 输出门 |
| Gated DeltaNet（30 层） | 32 V-head / 16 QK-head，head_dim 128，conv k=4，state fp32 | 线性注意力 + 卷积状态 + 循环状态缓存 |
| MoE | 256 专家，top-8 routed + 1 shared，intermediate **512** | softmax + top-k 重归一化，无 bias、无 routed_scaling |
| 位置编码 | `mrope_interleaved=true`，`mrope_section=[11,11,10]` | 3 段 mrope（多模态 RoPE） |
| MTP | `mtp_num_hidden_layers=1`，多步训练 | 投机解码可用（vLLM: `qwen3_next_mtp`） |
| 上下文 | 262,144 原生；YaRN factor 4 → 1,010,000 | |
| 权重 | 26 个 shard，合计 **71.9 GB** bf16；仓库内**没有 modeling 代码**，靠 transformers ≥4.57 原生实现 | 无量化变体，无量纲表 |
| 多模态 | `image_token_id=248056`，`video_token_id=248057` | 是 VL 模型，纯文本可 `--language-model-only` |
| 许可 | Apache-2.0 | |

## 2. DeepInfer 现在的形状

### 2.1 只认识一个架构

`deepinfer/model_executor/models/__init__.py:18-20`：

```python
_MODELS = {"DeepseekV41ForCausalLM": ("deepseek_vl", "DeepseekVLForCausalLM")}
```

`hugging_face/config.py:241-262` 是硬门禁，逐条不过直接 `ValueError`：

- `model_type` 必须 `== "deepseek_v41"`
- 必须有 `quantization_config`，且 `quant_method=fp8 / expert_dtype=fp4 / weight_block_size=[32,32] / scale_fmt=ue8m0`
- `num_key_value_heads` 必须 `== 1`（MLA 的 latent KV，注释原话 "one shared KV head"）
- 还要求二十余个 DeepSeek 专属字段：`compress_ratios`、`index_*`、`engram_*`、`dspark_*`、`hc_*`、`swiglu_limit`、`q_lora_rank`、`o_lora_rank`、`o_groups`、`sliding_window`…

Qwen3.6 的 config.json 里这些字段**一个都没有**，反向也一样。

### 2.2 注意力只有 FlashMLA

`layers/attention_backend.py:1-24` 只导出 `flash_mla_with_kvcache` / `flash_mla_sparse_fwd`（`vendor/FlashMLA` + 自维护 SM90 版），
MQA-latent 语义。全仓库 `grep flashinfer|paged_attention|paged_attn` **0 命中**；`flash-attn-4` 只被
`models/vision_encoder/vit.py:13` 的 ViT 用到，LM 侧不用。

### 2.3 MoE 门控是 DeepSeek 的

`v4_modules/moe.py:41` 原话："Scoring uses **sqrtsoftplus**; top-k selection uses **noaux_tc** plus a load-correction bias"。
Qwen3.6 是 softmax + top-k 重归一化，没有 bias，也没有 `routed_scaling_factor`。
底层 `sm90_ops/fused_moe_fp8.py` 是 block-FP8 W8A8（128×128 权重块 / 1×128 激活组）Triton GEMM，
SM90 默认路由专家核是 `humming-kernels` 的 W4A8。**没有 bf16 权重路径**。

### 2.4 不存在的东西（grep 全仓库 0 命中）

`gated_delta / delta_rule / linear_attn / conv1d / mamba / ssm`、mrope、tensor parallel
（`config.py:150-183` 的 `ParallelConfig` 只有 `data/context/expert/engram` 四个并行维，激活全无 all-reduce）。

### 2.5 词表与协议层是钉死的

`hugging_face/tokenizer.py:34-40`：BOS 必须是 `<｜begin▁of▁sentence｜>`@0，EOS@1，PAD@2，多模态 span token 必须落在 129280–129300。
协议层是 `deepseek-recipe`（Rust `.so`），DeepSeek 的 prompt 组装 / thinking 标记 / 解析都在里面。

### 2.6 环境也差一档

容器内 `transformers 4.50.0`，而 Qwen3.6 的 config 声明 `transformers_version: 4.57.1`；
`AutoTokenizer` 要认 `qwen3_5_moe_text` 得起版本。升级会牵动 DeepSeek 分词器的既有行为。

### 2.7 能复用的部分

| 组件 | 行数 | 可用性 |
|---|---|---|
| scheduler / block manager（Rust `_C`） | — | 部分，但 `prepare_prompt_v4` / `prepare_decode_v4` 是 V4 元数据形状 |
| hicache 前缀缓存 | 466 | 基本可用，但要补「循环状态随前缀一起失效」的语义 |
| structure_output（JSON 约束解码） | 595 | 可用，模型无关 |
| sampling（fused_sample / random_sampling） | — | 可用 |
| HTTP + 三个协议面 | — | 可重写，但协议层要换成 Qwen 的 |

粗算可复用 ≈ 4–5k 行 / 全仓 50.6k 行，**< 10%**。

## 3. 差距清单与工作量

| 子系统 | 现状 | 要做的事 | 人日 | 风险 |
|---|---|---|---|---|
| 配置 / 注册 / 权重加载 | 单架构 + 硬门禁 | 新 config 解析器；40 层混合层表；256 专家与 GDN 参数的命名映射；bf16 无量化通路 | 8–12 | 低 |
| Gated Attention（10 层） | FlashMLA(MQA latent) | paged-KV GQA 核（hd 256、2 KV head、varlen prefill）；64 维部分 RoPE；output gate；3 段 mrope | 15–25 | 中 |
| Gated DeltaNet（30 层） | 完全没有 | chunked prefill 核 + recurrent decode + conv state + fp32 状态池 + 与前缀缓存的失效联动 | 25–40 | **高** |
| MoE | sqrtsoftplus/noaux_tc + FP8/FP4 | softmax+重归一化门控；bf16（或 FP8）grouped GEMM；shared expert 拆开；关掉 eplb | 10–18 | 中 |
| MTP | DFlash / nextn（DeepSeek 结构） | Qwen 的 MTP head + 调度对接（`mtp_use_dedicated_embeddings=false`） | 8–12 | 中 |
| Tokenizer / 协议 | deepseek-recipe + 词表硬校验 | Qwen 分词器 + `chat_template.jinja`（7.7 KB）+ thinking 分段 + tool-call 解析 + `<|im_end|>` 停止 | 6–10 | 低 |
| 数值对齐 / 测试 | — | 与 HF/vLLM 逐层 logits 对齐、长上下文、状态缓存回归 | 10–15 | 中 |
| **最小可用小计** | | 纯文本、单副本、无 MTP、不追性能 | **82–132** | |
| 多模态 | DeepSeek-VL 专用 | Qwen3-VL ViT + merger + mrope 对齐 | 20–35 | 中 |
| CP / 性能调优 | CP 只对 MLA 有意义 | 让 CP 支持 GQA 与 GDN 状态分片；吞吐调到能打 | 15–25 | 高 |
| **生产级小计** | | | **117–192** | |

两点说明：

- 若直接 vendor `flash-linear-attention`（Apache-2.0，有 `chunk_gated_delta_rule` 的 Triton 核）当参考，
  GDN 那 25–40 人日可压到 ~18，最小可用落到 **75–110 人日**。即便如此仍是「一个月以上、两个人」的量级。
- 这是**熟手**估算。GDN 的数值路径（fp32 状态、conv 边界、chunk 间递推）最容易翻车，排期建议再乘 1.5。

## 4. 收益侧：为什么这笔投资不划算

**装得下，不需要任何并行花招。** 权重 71.9 GB + 256K 上下文的 KV ≈ 5.2 GB（10 个全注意力层 × 2 KV head × 256 维 × 2 × 2 B
= 20.5 KB/token），合计 **77 GB < 单卡 141 GB**。8 张卡就是 8 个副本，或者 TP8 把首 token 延迟压下来。

**带宽账。** 单副本 decode 每步读 72 GB，H20 约 4.0 TB/s，≈ 18 ms/token，bs=1 约 55 tok/s；
批量下这一次读摊给整批，吞吐线性上去。DeepInfer 想赢只能靠把专家权重压到 FP4（4× 带宽），
但那要自建量化管线 + 精度评估（另 3–5 人周），且 vLLM 的 NVFP4 路线已经在推。

**它的看家本领在这里全部空转。**

| DeepInfer 的强项 | 对 DeepSeek | 对 Qwen3.6-35B-A3B |
|---|---|---|
| MLA + 压缩 KV | 1M 上下文的关键 | 模型不用 MLA，KV 只有 20.5 KB/token，5.2 GB 就够 256K |
| FP8 权重 + FP4 专家 | 671B 装进 8 卡的前提 | 官方只有 bf16；能压但要自己做，且精度要重验 |
| Engram / yoco / HC | 稀疏记忆与跨层复用 | 模型里没有对应结构 |
| CP = 8 | 把 1M 上下文切成 8 份 | KV 总量太小，切了只增加通信 |

**替代路径已经就绪。** 模型卡本身就给了 vLLM / SGLang 的官方命令，含 NEXTN 投机解码
（`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`）、
`--reasoning-parser qwen3`、`--tool-call-parser qwen3_coder`。落地成本 1–2 天，且是社区共同维护。

## 5. 如果仍然要做：分阶段与止损点

**Phase 0（10 人日，先做，便宜且能一票否决）**

1. 单卡纯 PyTorch 参考实现（GDN + gated attn + MoE + MTP），与 HF/vLLM 对齐首 token 与 top-5 logits；
2. 同一份权重上量 vLLM 的 decode 吞吐，作为 baseline；
3. 估出 DeepInfer 改造后在同样硬件上的**上限**（只算 kernel 理论时间，不写全栈）。

Go 判据（三条全中就继续）：逐层 logits 对齐通过；DeepInfer 预估上限 ≥ vLLM 的 1.3×；且有明确业务理由必须留在 DeepInfer 栈内。
否则停在这里，沉没成本 10 人日。

**Phase 1（其余 70–120 人日）** 纯文本 MVP：配置/加载 → GQA 后端 → GDN → MoE → 分词与协议 → 对齐测试。
**Phase 2（另 40–70 人日）** MTP、多模态、CP、性能调优。

**止损点**：GDN 的 30 层如果 15 人日内拿不到与参考实现一致的逐层输出，说明这条路的技术风险已经超过收益，应止损。

## 6. 建议

1. **文本推理走 vLLM 或 SGLang**，和现有 new-api 网关对接。8 张 H20 上按 `--tensor-parallel-size 8`
   或 8 个 DP 副本部署，1–2 天可上线。
2. **DeepInfer 继续只服务 DeepSeek 系**。它是一台为单一架构做到极致的专用引擎，这正是它在 H20 上赢的原因；
   让它通用化会同时损失性能和可维护性。
3. 如果真实诉求是「一个入口、一套运维」，**统一在网关层，不要统一在引擎层**。new-api 已经把 27 号渠道
   （DeepInfer）和其它渠道放在同一个 OpenAI/Anthropic 协议面后面，再加一个 Qwen 渠道，架构上等价于「一个引擎」，
   成本接近 0。
4. 只有当出现「必须在 DeepInfer 的 CP / Engram / FP4 栈内跑 Qwen」的硬约束时才启动 Phase 0，
   并且把 10 人日当作可丢弃的探针预算。

## 附录：证据清单

```bash
# 目标模型
curl -sL https://hf-mirror.com/Qwen/Qwen3.6-35B-A3B/resolve/main/config.json
curl -sL https://hf-mirror.com/api/models/Qwen/Qwen3.6-35B-A3B/tree/main   # 26 shard, 71.9 GB
sed -n '35,60p' README.md                                                  # 层布局与各维

# DeepInfer
deepinfer/model_executor/models/__init__.py:18-20      # 唯一架构
deepinfer/hugging_face/config.py:241-262               # 硬门禁
deepinfer/layers/attention_backend.py:1-24             # FlashMLA only
deepinfer/models/v4_modules/moe.py:41                  # sqrtsoftplus + noaux_tc
deepinfer/layers/sm90_ops/fused_moe_fp8.py:1-20        # block-FP8 W8A8
deepinfer/hugging_face/tokenizer.py:34-40              # 词表硬校验
deepinfer/config.py:150-183                            # 无 TP
grep -rniE "gated_delta|delta_rule|linear_attn|conv1d|mamba|ssm" deepinfer/   # 0 命中
grep -rniE "flashinfer|paged_attention" deepinfer/                            # 0 命中
```
