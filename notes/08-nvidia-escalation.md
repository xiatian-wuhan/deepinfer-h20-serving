# 升级工单材料：H20-3e + R595 服务器驱动下 GPUDirect RDMA 不可用

提交对象：NVIDIA 企业支持 / 服务器厂商（浪潮同规格）
整理：2026-10-22 · 现场取证，全部命令可复现

---

## 一句话

在 **8× H20-3e + `nvidia-driver-595-server` 595.91.07 + Ubuntu 22.04 (kernel 6.8.0-57)**
上，GPUDirect RDMA 的两条实现路径（`nvidia-peermem` 与 dma-buf）**都不可用**：
`nvidia-peermem` 是空壳模块加载即 `EINVAL`；所有 dma-buf 相关 CUDA 设备属性都报不支持。
这使 NCCL 的 **GIN / GDAKI** 无法初始化，依赖它的 DeepEP Engram（DeepSeek-V4.1 的
n-gram 表）无法工作。

## 环境

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA H20-3e（`vbios 96.00.C3.00.02`） |
| 驱动 | `595.91.07` / `nvidia-driver-595-server` `595.91.07-0ubuntu0.22.04.1` |
| CUDA driver API | `13020`（CUDA 13.2） |
| OS / 内核 | Ubuntu 22.04 / `6.8.0-57-generic` |
| 启动 | UEFI；`/proc/cmdline` 含 `iommu=pt` |
| 网卡 | Mellanox `mlx5_0` **RoCE 100 Gb/s, ACTIVE**（`mlx5_1` DOWN）；`ens20f0np0 = <h20-host>` |
| NCCL | 2.32.3（`nvidia-nccl-cu13`） |
| rdma-core | 54（自建，确认含 `mlx5dv_reg_dmabuf_mr@MLX5_1.25`） |
| 容器 | `--network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1` |

容器内 NCCL 确实拿到了 RDMA 设备：

```
NCCL INFO NET/IB: [0] mlx5_0:uverbs0:1/RoCE provider=Mlx5 speed=100000
NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [RO]; OOB ens20f0np0:<h20-host><0>
NCCL INFO Initialized NET plugin IB
```

## 现象

NCCL 初始化 GIN 时（`src/transport/net_ib/gin.cc`，8 个 rank 全部一致）：

```
ncclGinIbGdrGpuSupport (gin.cc:42) NCCL WARN
  Unable to use GIN: Peermem is not supported, and device N does not support DMA-BUF.
```

对应 NCCL 源码的判定逻辑：

```c
bool peerMemSupport = gdaki ? ncclIbPeerMemSupport() == ncclSuccess : ncclIbGdrSupport() == ncclSuccess;
if (peerMemSupport) return ncclSuccess;
CUCHECK(cuDeviceGetAttribute(&dmaBufSupportOnDevice, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cudaDev));
if (dmaBufSupportOnDevice == 1) return ncclSuccess;
WARN("Unable to use GIN: Peermem is not supported, and device %d does not support DMA-BUF.", cudaDev);
```

## 取证 1：CUDA 设备属性整族都报不支持

```python
cu.cuDeviceGetAttribute(124, dev)   # CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
# -> (CUDA_SUCCESS, 0)
cu.cuDeviceGetAttribute(146, dev)   # CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED
# -> (CUDA_SUCCESS, 0)
cu.cuDeviceGetAttribute(152, dev)   # CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED
# -> (CUDA_ERROR_INVALID_VALUE, None)
cu.cuDeviceGetAttribute(115, dev)   # CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED
# -> (CUDA_SUCCESS, 1)      # 说明属性查询通道本身正常，不是全 0
```

## 取证 2：IOMMU 已经是 passthrough，不是它的问题

```
[   24.463232] iommu: Default domain type: Passthrough (set via kernel command line)
```

（此前是 `Translated`。改成 `iommu=pt` 并重启后，上述属性**没有任何变化**。）

内核 dma-buf 支持齐备：

```
CONFIG_DMABUF_MOVE_NOTIFY=y
CONFIG_DMABUF_HEAPS=y
CONFIG_UDMABUF=y
```

`/dev/dri/card0` … `card7` 均在；`nvidia.ko` 内含 `dma_buf` 相关字符串 12 处。

## 取证 3：`nvidia-peermem` 是空壳模块，加载必然失败

```
$ modprobe nvidia_peermem
modprobe: ERROR: could not insert 'nvidia_peermem': Invalid argument
$ dmesg | tail
（内核完全没有输出 —— 在模块 init 之前就被拒绝）
```

```
$ ls -l /lib/modules/6.8.0-57-generic/updates/dkms/nvidia-peermem.ko
-rw-r--r-- 1 root root 7904  nvidia-peermem.ko        # 仅 7.9 KB
$ nm -u nvidia-peermem.ko
                 U __fentry__
                 U param_ops_int
                 U __x86_return_thunk                  # 不引用任何 nvidia_p2p_*
$ grep -c nvidia_p2p_get_pages /proc/kallsyms
0                                                       # nvidia.ko 未导出该符号
```

即：真实 peermem 依赖的 `nvidia_p2p_get_pages` / `nvidia_p2p_dma_map_pages` 等
**在 R595 上不再导出**，随包附带的 `nvidia-peermem.ko` 没有任何 p2p 依赖，
是一个无法工作的占位模块。

## 我们已经排除的

| 假设 | 结论 |
|---|---|
| 容器看不到 IB 设备 | 已排除：`--network host --device /dev/infiniband` 后 NCCL 正常使用 `mlx5_0:1/RoCE` |
| rdma-core 太旧，缺 `mlx5dv_reg_dmabuf_mr@MLX5_1.25` | 已排除：升级到 rdma-core 54 后 `dlvsym` 报错消失 |
| IOMMU 严格模式阻断 P2P | 已排除：改 `iommu=pt` 并重启，属性仍为 0 |
| CUDA 头文件里该属性已废弃、驱动可能换了语义 | 已排除：14x/15x 同族属性也全报 0，而 115 等属性正常返回 |
| DeepEP/NCCL 用法错误 | 已排除：用 DeepEP 官方 `tests/engram/test_engram.py` 与最小 NCCL 建链均可复现 |

## 我们的问题

1. **H20-3e + R595 是否支持 GPUDirect RDMA？** 若支持，走哪条实现（dma-buf 还是 peermem）？
2. 在 `iommu=pt` + `CONFIG_DMABUF_MOVE_NOTIFY=y` + `/dev/dri/card*` 齐备的前提下，
   为什么 `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` 仍为 **0**？
3. 是否需要特定的 BIOS 设置（PCIe ACS / Above 4G Decoding / Resizable BAR / VT-d）？
   本机 `dmesg` 有 `DMAR: IOMMU feature prs inconsistent` 告警，是否相关？
4. 是否存在驱动级开关（`NVreg_*`）用于启用 dma-buf 导出？
5. **哪个驱动版本在 H20-3e 上是 GPUDirect RDMA 的验证过的组合？**
   我们受限于生产负载（vLLM 的 torch 为 `2.13.0+cu132`）不能低于 590。
6. 确认 `nvidia-peermem` 在 R595 上是否为有意退役的占位模块、
   `nvidia_p2p_*` 是否已不再导出。

## 为什么这件事阻塞我们

DeepSeek-V4.1-Flash-1017 的 n-gram 嵌入表由 DeepEP Engram 承载，
`EngramBuffer.fetch` 直接启动 GIN RDMA 内核（形参含 `ncclWindow_vidmem*` / `ncclGinRequest_t*`）。
GIN 不可用时该内核拿到空上下文，首次前向即在权重加载后的 warmup 阶段
`CUDA_ERROR_ILLEGAL_ADDRESS`（sticky error 最终暴露在后续 `cuModuleLoadData`）。

且 `vendor/DeepEP/csrc/kernels/comm/context.cpp` 硬编码
`reqs.ginType = NCCL_GIN_TYPE_GDAKI` 并断言 `props.ginSupport[NCCL_GIN_TYPE_GDAKI]`，
所以 **`NCCL_GIN_TYPE=PROXY` 也无法绕过**（已实测）。
参考部署文档亦把 GDR 当作既定前提（`doc/inference_deployment_qa.md:157`：KV swap-in 用 GDR 实现）。

## 需要的产出

一个「H20-3e 上 GPUDirect RDMA 可用」的确认 + 具体配置清单（驱动版本、BIOS 项、内核参数），
或明确告知该 SKU 不支持、我们改走多节点方案。
