# DGX Spark（GB10）上 ≥27B 模型的 prefill / decode 选型

调研 2026-10-23。每条数字标了来源、负载与投机配置；没有实测的写"未实测"。

## 0. 结论

| 指标 | 冠军 | 实测 |
|---|---|---|
| **单流 decode** | **Qwen3.6-35B-A3B + NVIDIA NVFP4 + vLLM + MTP k=3** | **108.3 tok/s**（裸奔 76.7） |
| **单机并发** | 同上，8 并发 | **375.8 tok/s** 聚合 |
| **prefill（dense 27B 基线）** | Qwen3.8-27B NVFP4 @8K 输入 | 2,635 tok/s（8K）→1,274（120K） |
| **prefill（MoE）** | Qwen3.6-35B-A3B，双机 TP=2 | 7,943 tok/s（短输入，**单机未实测**） |
| 框架 | **vLLM**（生态+投机解码最全），SGLang 次之但单流可以打到 78.6 | 见 §5 |

一句话：**在 ≥27B 这一档，Qwen3.6-35B-A3B 这一类的「小激活 MoE」同时赢了 prefill 和 decode**，
原因只有一条 —— 它每 token 只读 3B 的权重。这不是调优问题，是物理。

## 1. 一条公式解释全部现象

GB10：aarch64 / sm_121 / **128 GB 统一内存（可用约 121 GB）** / FP4 1 PFLOP（稀疏）/ **约 273 GB/s** / 约 140 W。

```
单流 decode ≈ 273 GB/s ÷ 每 token 读取的字节数 ≈ 273 GB/s ÷ 激活参数字节数
```

| 模型 | 每 token 需读 | 带宽账上限 | 实测裸奔 | 兑现率 |
|---|---:|---:|---:|---:|
| Qwen3.6-27B（dense，足迹 20 GB） | ~20 GB | 13.6 | **12.43** | 91% |
| Nemotron 3 Super 120B-A12B（激活 12B） | ~7 GB | 39 | **15.77** | 40% |
| Qwen3.6-35B-A3B（激活 3B） | ~2 GB | 136 | **76.66** | 56% |

读法：dense 27B 已经贴着带宽线跑（91%），没有余量；两个 MoE 只兑现了四到五成带宽——
专家权重是散落读取的，小 GEMM 吃不满。**但它们的天花板高 3–10 倍，所以就算只发挥一半也赢。**

顺带得到一个证伪工具：「dense 27B 在 Spark 上裸跑超过 13.6 tok/s」的宣称可以直接判假
（要用投机解码才行，那是另一回事）。

prefill 是算力账：dense 27B 在 8K 输入下 2,635 tok/s ≈ 13% MFU。MoE 把这条账按激活比例缩小，
所以 35B-A3B 的 prefill 能到 dense 的 3 倍。

## 2. 单机单流 decode 实测排行

| 模型 | 量化 | 框架 | 投机 | decode tok/s | 来源 |
|---|---|---|---|---:|---|
| **Qwen3.6-35B-A3B** | NVFP4（NVIDIA ModelOpt） | vLLM 0.25.0 | **MTP k=3** | **108.3** | DevelopersIO |
| 同上 | 同上 | 同上 | MTP k=2 | 98.0 | 同上 |
| 同上 | 同上 | 同上 | 无 | 76.7 | 同上 |
| 同上 | NVFP4（Unsloth 动态 W4A4） | 同上 | MTP k=3 / k=2 / 无 | 89.9 / 81.6 / 67.5 | 同上 |
| Qwen3.8-27B | NVFP4 | SGLang + DFlash2 | draft=16 | 78.6（77.0–78.9, n=5） | darkdatter |
| 同上 | 同上 | 同上 | draft=8 | 61.3 | 同上 |
| Qwen3.6-27B（dense） | NVFP4 | vLLM 0.25.0 | MTP k=3 | 28.9 | DevelopersIO |
| 同上 | NVFP4 | vLLM 0.25.0 | 无 | 12.43 | 同上 |
| Nemotron 3 Super 120B-A12B | NVFP4 | vLLM | 无 / MTP | 15.77 / 24.02 | 同上 |
| Qwen3.8-27B | FP8 | vLLM | 无 | 7.18 | aliyun |
| Qwen3.8-27B | GGUF Q4_K_M | llama.cpp | MTP n=5 | 21.44 | aliyun |

**同样是 27B 级，从 12.4 到 108.3 有 8.7 倍差距**，来源拆开是：激活参数（3B vs 27B）值 6 倍，
投机解码再值 1.4 倍。框架、量化的影响都在 ±20% 以内。

## 3. prefill 实测

| 模型 | 硬件 | 输入长度 | prefill tok/s | 来源 |
|---|---|---:|---:|---|
| Qwen3.8-27B | 1 机 | 8,192 | **2,635** | NVIDIA |
| 同上 | 1 机 | 9,698 | 2,173 | darkdatter |
| 同上 | 1 机 | 2,448 | 1,591 | 同上 |
| 同上 | 1 机 | 38,698 | 1,895 | 同上 |
| 同上 | 1 机 | 120,865 | 1,274 | 同上 |
| 同上 | 1 机 | 100K | 805 / 1,031（vLLM） | aliyun |
| 同上 | 1 机 | 100K | **1,106**（SGLang，chunk 8192） | aliyun |
| Qwen3.8-27B | 1 机 | 100K | 520–620（llama.cpp） | aliyun |
| Qwen3.6-35B-A3B | 2 机 TP=2 | 短 | **7,943** | lfendy |
| 同上（NVIDIA 权重） | 2 机 TP=2 | 短 | 6,915 | 同上 |
| Qwen3.6-35B-A3B | 1 机 | 短 | 3.5–4.5K（**折算，未实测**） | — |
| Nemotron 3 Super 120B-A12B | 1 机 | 短→长 | 140 → 1,900（随 prompt 增长） | vLLM 官方 |

三条可用结论：

1. **prefill 在 ~10K 见顶，然后随长度衰减**（2,173 → 1,895 → 1,274）。120K 输入要 95 秒才出第一个 token，
   这是 Spark 上做长上下文最硬的成本。
2. **短 prompt 区间反过来被固定开销吃掉**：vLLM 官方在 120B 上测到 prefill 从 140 tok/s（短 prompt）
   爬到近 1,900（长 prompt）。所以「prefill tok/s」这个数**不锁死输入长度就没有意义**。
3. **MoE 的 prefill 是 dense 的 3 倍左右**（7.9K vs 2.6K，虽不是同机同条件，但量级可信）。
4. **加机器对 prefill 帮助极小**：8 机只有 1.94×（NVIDIA 官方数据），而 decode 能到 4.43×。

## 4. 并发

| 模型 | 硬件 | 8 并发聚合 | 16 并发聚合 | 来源 |
|---|---|---:|---:|---|
| Qwen3.6-35B-A3B + MTP k=3 | 1 机 | **375.8** | — | DevelopersIO |
| Qwen3.6-35B-A3B 裸 | 1 机 | 298.7 | — | 同上 |
| Qwen3.8-27B + DFlash2 | 1 机 | 315.9 | **480.7** | darkdatter |

**并发是被配置卡住的，不是被硬件卡住的。** darkdatter 那台机器出厂配置只能跑 4 并发（190 tok/s），
三个 flag 同时限制（`max-running-requests`、`max-mamba-cache-size`、`cuda-graph-max-bs-decode`），
放开后 8→16 流从 190 涨到 480。只改一个量不出效果。

但「并发越高越好」也不成立：vLLM 官方在 Spark 上的建议是 `--max-num-seqs 4`，理由是超过 4 路解码后
每 token 的带宽税会盖过连续批处理的收益、TTFT 开始劣化。两者的差别在模型规模——27B 能吃到 16 路，
120B 在 4 路就饱和。**并发甜点随模型变，别照抄别人的数字。**

## 5. 框架矩阵（2026-10）

| 框架 | sm_121 状态 | 量化 | 投机解码 | 评价 |
|---|---|---|---|---|
| **vLLM** | v0.32.0（2026-10-18），官方镜像 `nvcr.io/nvidia/vllm:26.10.pre1-py3`，arm64 齐全 | NVFP4/FP8/AWQ/GPTQ/GGUF/Marlin | `--speculative-config`：mtp/eagle3/ngram/dflash/DSpark/draft_model | **首选**。投机解码最全，35B-A3B 的 MTP 开箱即用 |
| **SGLang** | v0.5.22（2026-10-16），`lmsysorg/sglang:v0.5.22-cu130` 有 arm64 | NVFP4(`modelopt_fp4`)/FP8/MXFP4 | NEXTN/EAGLE3/DFLASH/UNGRAM/STANDALONE | 单流能打到 78.6（靠 DFlash2），但要配 flag 才不被卡在 4 并发 |
| TensorRT-LLM | 1.3.0，官方钉 **rc12/rc13** | FP8/NVFP4/MXFP4 | EAGLE-3/MTP | 官方支持最好但版本极敏感：rc15 在 GB10 双重回归（冷启动挂死 #14500、MTP 路径 ptxas 拒 PTX #14575） |
| llama.cpp | 自建 `-DCMAKE_CUDA_ARCHITECTURES=121a-real` | GGUF Q4_K_M / UD-Q4_K_XL | `--spec-type draft-mtp` | 最干净的原生路径，但速度只有 1/3 |
| NIM + ModelOpt | `*-dgx-spark` 变体 | NVFP4/W4A16/BF16 | 未查到 | 最省心的商用路径 |
| MLC-LLM | 查不到任何 GB10 支持 | — | — | 不建议评估 |
| gbx-lm | v0.8.2 起支持 Spark | 自研 ternary/1.6-bit | 自带 MTP head | 27B 只要 5.9GB，62 tok/s；但闭源二进制、生态窄 |

### 五个会白白烧掉几天的坑

1. **`FLASHINFER_CUDA_ARCH_LIST="12.1f"`** —— 必须是 family 模式，写 `12.1` 或 `12.1a` 都失败（FP4 的 `ldmatrix.b8x16.b4x16_p64` 只在 family mode 开）。
2. **vLLM 自建时 `TORCH_CUDA_ARCH_LIST` 不设** → 只含 sm_75 cubins，FP8 启动即崩（#49904）。
3. **SM121 硬限制**：cluster 仅 1×1×1、无 TMA multicast/DSMEM，因此 **TRT-LLM backend 与 DeepGEMM 在 SM121 完全不可用**；
   CUTLASS 只认 NVFP4 不认 MXFP4（MXFP4 要 cuDNN ≥9.14/9.15）。
4. **`--moe-backend flashinfer_b12x` 只在社区特供镜像上必要**。官方 vLLM 0.25.0 上实测 auto=77.52 / marlin=76.51 /
   b12x=77.67，差 1.5%（auto 内部就选 Marlin）。照着 Unsloth 的 "不开慢 2.5 倍" 去折腾官方镜像是白费。
5. **GB10 上别让 `gpu-memory-utilization` 超过 0.85–0.90**：统一内存会和 OS 抢，多台机器被记录到在内核加载阶段
   直接卡死（这也解释了为什么官方 35B-A3B 配方写的是 0.4）。

## 6. 量化选型：官方权重赢社区权重（和宣传相反）

| 权重 | 方案 | 单流 | 8 并发 |
|---|---|---:|---:|
| NVIDIA 官方 NVFP4 | ModelOpt；MoE 层 FP4 + 静态 FP8 缩放（unsloth 称之为 W4A16） | **76.66 / 98.0 / 108.3**（裸/MTP2/MTP3） | 298.7 / 357.4 / 375.8 |
| Unsloth NVFP4 | compressed-tensors；逐 token 动态量化激活（W4A4） | 67.52 / 81.6 / 89.9 | 264.9 / 361.4 / 344.5 |

Unsloth 宣传"比其它 NVFP4 快 2.5×"，那是 **B200 128 并发的数字**。同一台 Spark、同一 vLLM、同一 harness 下，
**官方权重反而快 12–17%**（两个版本 MTP 接受率几乎相同：0.87/0.69 vs 0.86/0.67/0.55，所以差距纯粹来自执行效率）。
8 并发时差距消失（W4A4 的逐 token 动态量化开销被摊平）。

第二个独立来源给出同样方向：**sm_120/121 上 W4A4 存在 grouped-FP4-GEMM 的失败模式，权重-only 的 W4A16 能绕开**。
所以在这台机器上，选官方权重不只是精度更稳，也是可用性更好。

## 7. 数据打架与不确定性

- **27B 单流 78.6 vs 28.9 vs 12.4**：三个都不是错。区别在 drafter（DFlash2 draft16 接受长度 9.52 ≫ MTP ~2.5）
  和上下文长度（aliyun 用 100K 输入）。**引用时不带这两个参数就没有可比性。**
- **35B-A3B 单机 108.3 vs AEON 的 45.7**：后者跑在 vLLM 0.22.1 的社区补丁镜像上，早了半年。以 0.25.0 的 108.3 为准。
- **单机 35B-A3B 的 prefill 没有实测数据**。3.5–4.5K 是我按双机 7.9K 折的，**不要当结论引用**。
- **8 机 prefill 只有 1.94×** 与"prefill 是计算密集"直觉冲突。NVIDIA 归因于 collective 开销，我没独立复现。

## 8. 复现（单机最短路径）

```bash
# 冠军配置：Qwen3.6-35B-A3B + NVIDIA NVFP4 + MTP k=3
docker run --gpus all --ipc=host --net=host vllm/vllm-openai:v0.25.0-aarch64 \
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --quantization modelopt --kv-cache-dtype fp8 \
    --max-model-len 65536 --max-num-seqs 8 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --enable-prefix-caching --reasoning-parser qwen3 --trust-remote-code

# 备选：Qwen3.8-27B + SGLang + DFlash2（单流 78.6，TTFT 190ms）
docker run --gpus all --ipc=host --net=host lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2 \
  python3 -m sglang.launch_server --model-path RadixArk/Qwen3.8-27B-NVFP4 \
    --speculative-algorithm DFLASH2 --speculative-draft-model z-lab/Qwen3.8-27B-DFlash2 \
    --speculative-num-draft-tokens 16 --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.85 \
    --max-running-requests 16 --max-mamba-cache-size 80 --cuda-graph-max-bs-decode 16
```

测法必须固定：prefill = prompt_tokens / TTFT；decode = (completion−1)/(末 token − 首 token)；
单流与并发分开报；**必须注明投机配置**。

## 9. 建议

1. **一台 Spark 装一个模型，选 Qwen3.6-35B-A3B + NVIDIA NVFP4 + vLLM + MTP k=3**。23 GB 足迹，
   108 tok/s 单流、375 tok/s @8 并发、262K 上下文、支持图像输入、Apache-2.0。这是当前 ≥27B 的默认答案。
   剩下 100 GB 统一内存可以再驻一个 RAG/ComfyUI，把 gpu-memory-utilization 压到 0.35 左右即可。
2. **只有在需要「更长的思维链质量」或已有 SGLang 运维时**才选 Qwen3.8-27B + DFlash2（78.6 但更吃定制镜像）。
3. **不要用 llama.cpp 做主力**，也不要为了「2.5× 的 NVFP4」去换 W4A4 权重——在 Spark 上两个都是负收益。
4. **双机只在装不下或要 120K+ 上下文时**才上：decode 1.84×、prefill 只有 1.25×。
5. 想要更强的模型而不是更快的：单机可跑 Nemotron 3 Super 120B-A12B（70 GB，15.8/24.0 tok/s）；
   235B 要双机；753B 要 8 机（8K 输入 1,600 tok/s prefill、67 tok/s @8 并发）。

## 附录：来源

1. DevelopersIO《A new champion among partner LLMs for DGX Spark?》— 单机 35B-A3B 两版量化的同条件实测
   <https://dev.classmethod.jp/en/articles/dgx-spark-qwen3-6-35b-a3b-nvfp4-new-champion/>
2. darkdatter/gb10-repo（Qwen3.8-27B + SGLang + DFlash2，含 RESULTS.md 方法学与 prefill 曲线）
   <https://github.com/darkdatter/gb10-repo>
3. NVIDIA《Running Frontier-Class AI Models Across 8 DGX Spark Nodes》，2026-10-22
   <https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/>
4. zorost/sparkduet RESEARCH.md（measured/reported/inference 三级标注）
   <https://raw.githubusercontent.com/zorost/sparkduet/refs/heads/model/deepseek-v4-flash-0731/docs/RESEARCH.md>
5. 阿里云《Qwen3.8-27B 部署测速报告（GB10 / DGX Spark）》，2026-08-17
   <https://developer.aliyun.com/article/1756323>
6. unsloth/Qwen3.6-35B-A3B-NVFP4-Fast discussions #4（双机 TP=2 的 pp/tg）
   <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast/discussions/4>
7. NVIDIA dgx-spark-playbooks（vllm / sglang / trt-llm / llama-cpp 四份官方配方）
   <https://github.com/NVIDIA/dgx-spark-playbooks>
8. nvidia/Qwen3.6-35B-A3B-NVFP4 模型卡（DGX Spark 推荐参数）
   <https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4>
9. vLLM 官方 DGX Spark 部署指南（统一内存、`max-num-seqs 4`、120B 五场景实测）
   <https://vllm.ai/blog/2026-06-01-vllm-dgx-spark>
