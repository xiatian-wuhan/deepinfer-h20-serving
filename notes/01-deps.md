# 依赖分析与镜像选型

## 构建入口（#92 的 build.sh 真实流程）

```
DEEPINFER_ACCELERATOR=nvidia ./build.sh
  1) pip install -r requirements-nvidia.txt
       └─ -r requirements-dev.txt → -r requirements.txt
       └─ torch==2.13.0+cu132   (--find-links https://download.pytorch.org/whl/cu132/torch/)
       └─ flash-attn-4==4.0.0b34
  2) cd deepinfer && python -c 'from deepinfer import …'  ← 通过源码直接跑起来，不 pip install
  3) source scripts/cuda_arch.sh
       └─ 若 DEEPINFER_CUDA_ARCH=sm90 → pip install -r requirements-sm90.txt
            └─ humming-kernels==0.1.12
  4) pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3
  5) bash scripts/build_vendor.sh     ← DeepSelect / FlashMLA-SM90 / DeepEP / DeepGEMM（含补丁）
  6) bash scripts/build.sh build      ← deepinfer._C
```

## 硬约束

| 项 | 要求 | 来源 |
|---|---|---|
| GCC | ≥ 13.0 | README |
| Python | 3.12 | README |
| CUDA | 13（作者实测 13.0–13.2） | README / PR#92 环境表 |
| NCCL | 2.32.3（build.sh 强制装） | build.sh |
| torch | 2.13.0+cu132 | requirements-nvidia.txt |
| GPU 架构 | sm90（H20 = SM90, compute_cap 9.0） | 已在本机 `nvidia-smi` 确认 |

## 两个 base 镜像候选

### A. `lmsysorg/sglang:v0.5.22`（本地已有，无需拉取）

实测内含：Python **3.12.15**、torch **2.14.1+cu130**、**CUDA 13.0**、nvcc 13.0、
g++ **13.3**、NCCL、flashinfer 0.7.1（含 sm90a cubin）、**humming-kernels 0.1.12**、
tilelang 0.1.12、`/usr/local/cuda-13.0/compat`（前向兼容 libcuda，正好应对 R590 驱动）

| 依赖 | 镜像现状 | DeepInfer 要求 | 处理 |
|---|---|---|---|
| Python | 3.12.15 | 3.12 | ✅ |
| g++ | 13.3 | ≥13 | ✅ |
| CUDA toolkit | 13.0 | 13.x | ✅（作者实测 13.0 可用） |
| NCCL | 有 | 2.32.3 | build.sh 会强制重装 |
| humming-kernels | **0.1.12** | 0.1.12 | ✅ 完全一致 |
| torch | 2.14.1+cu130 | **2.13.0+cu132** | ❌ 需降级 |
| tilelang | 0.1.12 | **0.1.15** | ❌ 需升级 |
| tile-kernels | 无 | 2.1.0 | ❌ 需安装 |
| flash-attn-4 | 未确认 | 4.0.0b34 | ❌ 需安装 |
| xgrammar | 未确认 | 0.2.2 | ❌ 需确认 |

**策略**：不要在镜像的系统 Python 里动 torch（会破坏 SGLang 自己的环境），
**新建独立 venv** `/opt/deepinfer/venv`，在里面装齐 DeepInfer 依赖。
CUDA toolkit / g++ 直接用镜像自带的。这样两套环境互不干扰。

### B. `nvidia/cuda:13.2-devel-ubuntu24.04`（需拉取 ~7 GB）

- 优点：CUDA 13.2 = cu132 轮子完全对齐，torch 版本无冲突
- 缺点：要拉 7 GB（4 MB/s ≈ 30 分钟），还要自己装 Python venv/pip

## 推荐路径

**先试 A**（省 30 分钟拉镜像）。若 torch 2.13.0+cu132 在 CUDA 13.0 toolkit 下编译
`deepinfer._C` 失败，再退到 B。

## 网络限制（重要）

本机出口带宽实测 **~4 MB/s 硬上限**（三源并行 / 32 并发都不叠加）。
torch cu132 轮子约 2–3 GB，单这一项就要 **10–15 分钟**。
整个镜像构建预计 **40–70 分钟**，其中下载占大头。

## 已就绪的本地资产

- 权重：`/home/ubuntu/data1/models/DeepSeek-V4.1-Flash-1017`（56 文件，已 Sha256 全量校验）
- 源码：`repos/` 下 5 个仓库，含 NVIDIA submodule（DeepSelect/FlashMLA/FlashMLA-SM90/DeepEP/DeepGEMM）
- docker 根目录 `/home/ubuntu/data0` 剩 **2.4 TB**
