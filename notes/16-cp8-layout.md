# 16 — CP8 复现：布局选错，不是代码慢

> **快照说明**：本文数字是 2026-10-22 在**专有内核模块 + `EP_DISABLE_GIN=1`** 下测的。
> 换开源驱动后同一 harness 重测：CP8 128K TTFT **3.859 s**（关 GIN 时 3.342）、c=32 **1614.3**、
> c=64 **1836.4**。当前口径见 `README §5.1`、`§5.3`。


日期：2026-10-22。主机 <h20-host>（8×H20-3e），commit `63b430a`，
镜像 `deepinfer-sm90:v41-sm90`，`EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1
DEEPINFER_USE_NEXTN_DECODE=5 DEEPINFER_ENABLE_MTP_VARLEN=1`，
`NCCL_NVLS_ENABLE=0`，`doc/sm90.md` 同款 yoco 设置。

## 起因

用户问：单会话解码只有 35.9-79.9 tok/s，参考 158；长输入预填 10.6 s，参考 2.1 s。
此前一直按 `--ep 8 --cp 1 --dp 8` 跑。`doc/sm90.md:59` 写的是
**解码 DP8×EP8 / 预填 CP8×EP8** —— 表里两种布局是两列，不是一种。

## 第一次尝试为什么失败

`--ep 8 --cp 8` 直接崩在 DeepEP 初始化：

```
RuntimeError: NCCL exception (vendor/DeepEP/csrc/kernels/comm/context.cpp:164):
5 (Unable to use GIN: Peermem is not supported, and device 4 does not support DMA-BUF.)
```

根因：`context.cpp:124` 只有 `get_env("EP_DISABLE_GIN", 0) != 0` 时才跳过
`ncclDevCommCreate` 的 GIN 申请。当时容器里已经把 `EP_DISABLE_GIN` 删掉了
（PR#99 之后以为不再需要），于是每个 rank 都在申请 GIN，而这台机的
nvidia-peermem 是 7.9 KB 空壳、dma-buf attr=0（见 notes/06、07），必然失败。

**注意这个开关是双关的**：`environ.py:155`
`DEEPINFER_ENGRAM_NVLINK_FETCH = EnvVar(lambda v: _bool(v, environ.get("EP_DISABLE_GIN","0")))`
——`EP_DISABLE_GIN=1` 同时（a）让 DeepEP 跳过 GIN，（b）打开 PR#99 的 NVLink
Engram 取数快路径。两者必须一起给，不能只给一个。这也是为什么
`DEEPINFER_ENGRAM_NVLINK_FETCH=0` 无法用来做 A/B：它会落回必须 GIN 的 C++
`EngramBuffer`，在这台机上跑不起来。

## 结果（`examples/latency_benchmark.py`，同机同 commit）

TTFT @128K（1 次预热 + 3 次测量，中位数）：

| 布局 | TTFT | vs 文档 |
|---|---:|---|
| EP8 DP8 CP1 | 10.641 s | 文档只公布 CP8 的 2.114 s，DP8 预填无公布值 |
| **EP8 DP1 CP8** | **3.064 s**（3.048 / 3.064 / 3.066） | 文档 CP8 = 2.114 s |

解码（1024 in / 512 out）：

| 并发 | DP8 总 tok/s | CP8 总 tok/s | DP8 单请求 | CP8 单请求 | DP8 TPOT 中位 | CP8 TPOT 中位 | DP8 TTFT 均值 | CP8 TTFT 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.3* | **199.1-231.8** | 44.3* | **~212** | 22.58 ms* | **4.31 ms** | — | 0.85-1.36 s |
| 32 | 1178.8 | **1720.9** | 68.6 | 62.6 | **12.97 ms** | 16.12 ms | 3.94 s | **2.41 s** |
| 64 | **2156.6** | 2097.8 | **57.0** | 39.9 | **16.23 ms** | 25.31 ms | 5.76 s | **3.78 s** |

\* DP8 c=1 那一列来自 `b-dec-dp8.log`，该跑前面先跑了 ttft 套件，128K 缓存把
KV 池打满，连 1024-token 输入的 TTFT 都到了 3.79 s（正常 ~0.2 s）。这组单会话
数字**不可信**，已另起一次同序干净对照（`dp8-clean.log`）复核。

## 怎么理解

- CP8 时 `--dp` 默认 `ep // cp = 1`，8 张卡协同服务**同一个** batch，等价于
  「注意力 TP8 + 专家 EP8」。DP8 是 8 个独立副本，一个请求只用**一张卡**的注意力。
- 所以 CP8 把每 rank 的注意力/KV 读取和 expert dispatch 量都降到 1/8，
  专家 GEMM 总量不变 —— 这解释了 3.5× 的 TTFT 改善和 ~5× 的单会话解码改善。
- 代价在饱和区：c=64 时 CP8 单请求 TPOT 25.3 ms 比 DP8 的 16.2 ms 差，
  聚合吞吐 2097.8 vs 2156.6（-2.7%，基本打平），但 CP8 的 TTFT 仍好 34%。
- `doc/inference_deployment_qa.md:47-49` 的官方口径其实是：
  **Prefill 用 EP2-8 / DP2-8，Decode 用 EP64-128 / DP64-128（Wide-EP）**，
  最小高效单元 64P + 64D。8 卡上不可能同时满足两边；README:42 也写明
  Prefill/Decode 分离**本版本未包含**。在单机 8 卡上，CP8×EP8 是更划算的那一个。

## 待查

- DP8 干净对照（`dp8-clean.log`）。
- accept 率差异：CP8 4.28-4.90，DP8 历史值 1.94-3.09。投机解码接受率理论上
  与并行方式无关，需要排除「Engram NVLink 取数在 DP8 下污染输出」或
  「缓存/套件顺序混淆」。干净对照出结果后判断。
- 残差 3.064 vs 文档 2.114 = 1.45×。已排除 NVLS（文档 SM90 默认就是 0，
  且 BucketBuffer 相比 NCCL all-gather 只差 0.005 s）。更可能是本机 GIN 缺失
  导致 DeepEP 走 NCCL 回退（notes/06/07/08 已定性为驱动侧死路）。

## 干净 A/B（`dp8-clean.log`，与 CP8 那跑逐字同序同环境，只差 `--cp/--dp`）

之前 `b-dec-dp8.log` 的单会话 44.3 tok/s、accept 1.94-3.09 是**缓存污染假象**
（同一进程先跑 ttft 套件，128K 条目打满 KV 池，1K 输入 TTFT 都到 3.79 s）。
重跑后 accept 恢复 4.59-4.88，与 CP8 的 4.28-4.90 一致 ——
**「Engram NVLink 污染输出」的假设被排除**。

| 指标 | DP8 干净 | CP8 | 参考 DP8（`doc/sm90.md`） |
|---|---:|---:|---:|
| TTFT @128K | 10.641 s | **3.064 s** | 2.114 s（这是 CP8） |
| c=1 解码 | 86.3 tok/s | **212** | 158 |
| c=1 TPOT | 11.5 ms | **4.3 ms** | 6.33 ms |
| c=32 总量 | 1615.7 | **1720.9** | 2912 |
| c=32 单请求 | 67.2 | 62.6 | — |
| c=32 TTFT 均值 | 4.60 s | **2.41 s** | — |
| c=64 总量 | **2770.6** | 2097.8 | 4259 |
| c=64 单请求 | **62.4** | 39.9 | — |
| c=64 TTFT 均值 | 6.81 s | **3.78 s** | — |

结论：

1. **CP8 复现成功**，且单会话解码 212 tok/s **超过参考 DP8 的 158**。
2. DP8 干净后仍差参考 1.8×，量级与实测量完全吻合：
   我们 54 ms/步（4.7 accept × 11.5 ms），参考 27 ms/步（4.3 × 6.33）；
   profiler 实测 Engram 纯冗余 33.1 ms/步，54 − 33 = 21 ms，落在参考水位附近。
   **这 1.8× 就是 `engram_nvlink.py::fetch` 的冗余，不是模型/硬件/驱动问题。**
3. 布局不是二选一的对错，而是负载形态：
   c≤32 用 CP8（延迟全面更优），c=64 高吞吐用 DP8（总量 +32%）。
   但两者都被同一个 Engram 冗余拖累，先修它收益最大且对两种布局都成立。
