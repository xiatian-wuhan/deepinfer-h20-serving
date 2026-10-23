# 18 — 1M 上下文该用哪种布局：CP8 vs DP8（实测）

> **快照说明**：本文数字是 2026-10-22 在**专有内核模块 + `EP_DISABLE_GIN=1`** 下测的。
> 换开源驱动后同一 harness 重测：1M TTFT CP8 **30.220 s** / DP8 **140.970 s**（关 GIN 时 CP8 25.274 s）。
> 当前口径见 `README §5.1`。


日期：2026-10-22。同机（8×H20-3e，141 GB/卡）、同 commit `63b430a`、同 env
（`EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1 DEEPINFER_USE_NEXTN_DECODE=5
DEEPINFER_ENABLE_MTP_VARLEN=1`），Engram fetch 已按 notes/17 修好。
`--max_model_len 1048576`。

## 结论先行

1. **CP8 支持 1M 上下文**：实测 1M prompt 成功处理，**TTFT 20.403 s**（参考 CP8 = 19.41 s，差 5%）。
2. **1M 不是显存问题**：实测 KV 为 **891 B/token**（`0.891 GiB / 8385 blocks / 128 tokens_per_block`），
   1M 上下文 ≈ **0.87 GiB**。两种布局在显存上都放得下，DP8 也一样。
3. **1M 是算力/延迟问题，而这里 CP8 是唯一实用的**：同一条 1M 请求
   **DP8 = 189.118 s，CP8 = 20.403 s，差 9.3×**。预填在 DP8 下只有一张卡算注意力，
   1M 时呈超线性恶化；CP8 把序列切 8 份，每 rank 只算 1/8。
4. 因此对「1M 上下文 + 并发 ≤32」这个需求，**CP8（`--ep 8 --cp 8`，DP 自动=1）是最优选择**。

## 完整对照（Engram 修好后）

| 指标 | CP8 | DP8 | 参考 |
|---|---:|---:|---:|
| TTFT @1M | **20.403 s** | 189.118 s | 19.41 s（CP8） |
| TTFT @128K | 3.064 s | 10.641 s | 2.114 s（CP8） |
| c=1 解码 | **240.6-279.1 tok/s** | 151.9 | 158（DP8） |
| c=1 TPOT | **3.58-4.16 ms** | 6.58 ms | 6.33 ms（DP8） |
| c=32 总吞吐 | 1777.1 tok/s | **2380.6** | 2912（DP8） |
| c=32 单请求 | 66.0 tok/s | **103.4** | — |
| c=32 TPOT | 15.25 ms | **9.70 ms** | — |
| c=32 TTFT | **2.30 s** | 3.69 s | — |

## 怎么理解这个取舍

- **CP8 = 注意力 TP8 + 专家 EP8**：8 张卡服务同一个 batch，每 rank 只处理 1/8 序列。
  单请求延迟、长上下文预填全面占优（1M 时 9.3×）。
- **DP8 = 8 个独立副本**：一个请求只吃一张卡的注意力，但 8 个副本各自成批，
  c≥32 时聚合吞吐更高（+34%），单请求 TPOT 也更好（9.70 vs 15.25 ms）。
- 分界点大约在 **并发 ~4-8**：低并发 CP8 全面赢，高并发 DP8 吞吐赢。
- 但**只要服务承诺 1M 上下文，DP8 就被排除**——189 s 的 TTFT 无法交付。

## 待办

- CP8 下 `max_num_seqs` 必须 ≥ 并发目标（benchmark 按每引擎在飞数自动规划；
  生产要在 serve 层显式给 `--max-num-seqs 32`）。
- `serve/deepseek_server.py` 目前只透传 `--dp`（= EP size），**没有 `--cp`**，
  要跑 CP8 必须补上（`num_engines = ep // cp`，`create_llm(cp_size=...)` 已支持）。
- CP4×DP2 这个中间点未测，若将来并发需求上移可以补。

## serve 层已验证（CP8 端到端）

`serve/deepseek_server.py` 已补 `--cp`：`--dp` 保持 "EP size = 用几张卡" 的语义，
引擎数 = `dp // cp`，`create_llm(cp_size=...)` 本来就支持，路由/健康检查/负载统计全部改成按引擎数。

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DEEPINFER_HF_MODEL_DIR=/model \
EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1 \
DEEPINFER_USE_NEXTN_DECODE=5 DEEPINFER_ENABLE_MTP_VARLEN=1 \
serve/server.sh start --dp 8 --cp 8 --max-num-seqs 32 --port 8888
```

日志确认 `cp_rank=0..7`、`engine 0 ready (1/1)`、`/health` 返回 `ready_ranks:1`。
`tools/serve_test.py` 在 CP8 下 **14 passed, 0 failed**，含流式 SSE、
`usage.prompt_tokens_details.cached_tokens`、以及同 prompt 二次请求
`cached_tokens=3840/4019`（前缀缓存生效）。

注意 `--max-num-seqs`：CP8 只有一个引擎，它直接决定整机并发上限，
要跑 32 并发就必须显式给 `--max-num-seqs 32`（默认 128 也没问题，只是显存按需）。

## 图片输入（OpenAI 多模态）

`serve/deepseek_server.py` 已支持图片。契约来自模型自带的 `encoding/encoding.py`：
`encode_messages(..., return_multi_modal_data=True)` 返回 `(prompt, {"images": records})`，
每个图在 prompt 里落一个 `<｜deepseek_image｜>`（id 129264），引擎的
`multi_modal_processor.overwrite_prompt` 再按 `compute_image_layout` 把它展开成
1 个 image token + N 个 pad token。所以 serve 只需给出 `image_sizes=[(w,h)]` 与 `image_bytes=[...]`。

实测（模型自带样图 `inference/examples/images/corn.jpeg`，450×308）：

| 用例 | 结果 |
|---|---|
| 单图 base64 data URL | 200，正确描述"三根玉米、中间一根剥开露出金黄色玉米粒、纯白背景" |
| 同图第二次 | 0.53 s |
| 两张图同请求 | 200，正确回答"2"，`prompt_tokens=394` |
| 流式 + 图 | 正常，usage 含 `prompt_tokens=198` |
| Anthropic `image`+`source.base64` | 200，答"玉米" |
| http URL（默认） | 400，明确提示用 data URL 或开 `--allow-image-urls` |

token 记账已被验证为精确：450×308 触发 `min_pixels=295936` 放大到 657×450，
grid `(11,16)` → `11*(16+1)+2 = 189` 个图像 token，加文本 14 = **203**，与返回的
`prompt_tokens` 一致。`_image_token_count` 是 `compute_image_layout` 的逐行移植，
对 **410 个尺寸**（含极端长宽比）与引擎实现比对 **0 处不一致**。

默认只允许内联 base64：该端点没有鉴权，放开任意 URL 等于给外部一个 SSRF 代理。
