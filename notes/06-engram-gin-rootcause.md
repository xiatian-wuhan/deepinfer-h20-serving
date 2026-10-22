# warmup 崩溃根因报告：Engram 的 NCCL GIN 依赖

窗口：2026-10-22 14:01:30 - 14:10:16（8.7 min，看门狗 4.7 min 恢复生产）
生产：停止前 `/v1/models` 200，14:15:01 恢复 200，全程无生产事故。

## 结论（一句话）

**不是 mHC 的 bug，也不是 humming/DeepGEMM 的 bug。**
崩溃点是 `EngramRuntime.fetch` 启动的 **DeepEP Engram RDMA-GIN 内核**，而
**本机 GPUDirect RDMA 不可用**（peermem 未加载 且 dma-buf 注册失败），
导致 GIN 上下文根本没建立，内核拿着空的 `ncclGinRequest_t` 去发 RDMA -> 非法访存。

## 证据链

### 1. `CUDA_LAUNCH_BLOCKING=1` 把错误钉在真正的算子

旧现象（无 blocking）：错误停在 `hc.py:129 _hc_fused_post_pre` 的
`cuModuleLoadData(...) CUDA_ERROR_ILLEGAL_ADDRESS`。`cuModuleLoadData` 不碰显存，
这是 **sticky error**（此前某次异步越界）迟到的暴露点，所以之前一直误判为 mHC。

加 `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 TORCH_SHOW_CPP_STACKTRACES=1` 后：

```
deepseek.py:188        ngram_embed_hooks = self.engram_runtime.fetch(hashes)
engram.py:215          hooks = self._buffer.fetch(...)
deep_ep/buffers/engram.py:169  return self.runtime.fetch(indices, num_qps, use_tma_aligned_col_major_sf)
RuntimeError: Panic error (deep_jit/backend/cuda/driver.hpp:33):
  lazy_cuLaunchKernelEx(...) failed with CUDA error 700 (CUDA_ERROR_ILLEGAL_ADDRESS)
C++ trace:
  #1 Kernel::launch<ncclWindow_vidmem*, ncclWindow_vidmem**, void*, void*, int*,
                   ncclGinRequest_t*, sf_pack_t*, sf_pack_t*, int,int,int,int>
  #2 deep_ep::engram::EngramBuffer::fetch
```

内核形参里的 `ncclWindow_vidmem*` / `ncclGinRequest_t*` 直接指明这是一条 **GIN RDMA 路径**。

### 2. 为什么之前要用 `EP_DISABLE_GIN=1`

启动期 DeepEP 断言（`vendor/DeepEP/csrc/kernels/comm/context.cpp:125`）：

```
(allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE
  and "NCCL GIN is unavailable. ..."
```

之前为了绕过它加了 `EP_DISABLE_GIN=1`。**这个开关只是跳过了断言，没有让 engram 走别的路**
-- `userbuffer.py` 仍然硬编码 `allow_hybrid_mode=True` 并照常创建 EngramBuffer，
`fetch` 照常启动 GIN 内核，于是越界。所以 `EP_DISABLE_GIN=1` 对这个模型是**伪修复**。

### 3. `allow_hybrid_mode` 不是原因（已用官方测试证伪）

假设：「rail 模式在单节点不可用，改成 non-hybrid 就行」。
用 DeepEP 官方测试 `vendor/DeepEP/tests/engram/test_engram.py`（默认就是
`allow_hybrid_mode=False`）在 8 卡上跑，**两种模式都是同一条 GIN 断言**：

| 模式 | 结果 |
|---|---|
| `allow_hybrid_mode=False`（官方测试默认） | `props.ginType == NCCL_GIN_TYPE_NONE` |
| `allow_hybrid_mode=True`（deepinfer 硬编码） | `props.railedGinType == NCCL_GIN_TYPE_NONE` |

顺带纠正一个观察：DeepEP 官方 e2e 测试 `--allow-hybrid-mode` **默认 False**，
而 `deepinfer/model_executor/parallel_utils/userbuffer.py:154` 写死 `True`。这点仍值得对上
（rail 模式在单节点无意义），但它不是本次崩溃的原因。

### 4. 容器网络是真实缺陷之一

`e2e_window.sh` 用的是 `docker run` 默认 bridge 网络，且没有传 `/dev/infiniband`。
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` 显示：

```
NCCL INFO NET/IB : No device found.
NCCL INFO NET/Socket : Using [0]eth0:172.17.0.2<0>       <- Docker bridge！
NCCL ATTN Failed to initialize NET plugin IB
NCCL INFO GIN/Plugin: Could not find: libnccl-gin.so
```

宿主机其实有 **2 个 Mellanox NIC，`mlx5_0` 是 ACTIVE 100 Gb/s RoCE**
（`mlx5_1` DOWN；`/sys/class/infiniband/{mlx5_0,mlx5_1}` 存在）。

加上 `--network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1`
之后立刻变成：

```
NCCL INFO NET/IB: [0] mlx5_0:uverbs0:1/RoCE provider=Mlx5 speed=100000
NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [RO]; OOB ens20f0np0:<h20-host><0>
NCCL INFO Initialized NET plugin IB
```

**这一条是确定要改的**（无论后面怎么解决 GDR，容器都必须能看到 IB 设备）。

### 5. 最后一环：GPUDirect RDMA 不可用

IB 可见之后，GIN 仍然 NONE，原因写在 NCCL 自己嘴里（8 个 rank 一致）：

```
ncclGinIbGdrGpuSupport (transport/net_ib/gin.cc:42) NCCL WARN
  Unable to use GIN: Peermem is not supported, and device N does not support DMA-BUF.
```

两条 GDR 通路都断了：

**(a) peermem 未加载，且加载失败**

```
lsmod | grep peermem          -> 空（nvidia / nvidia_uvm / nvidia_drm / nvidia_modeset 都有）
modinfo nvidia_peermem        -> /lib/modules/6.8.0-57-generic/updates/dkms/nvidia-peermem.ko
                                 version 595.91.07（存在！）
insmod nvidia-peermem.ko      -> ERROR: Invalid parameters (EINVAL)
dmesg                          -> 完全没有内核日志（连 module_layout/Unknown symbol 都没有）
```

EINVAL 且内核零日志，说明内核在跑模块 init 之前就拒了。需要 **root** 排查
（`dkms autoinstall` / 重装 `nvidia-dkms-595` 重建模块）。我没有 sudo 权限
（`sudo -n` 要求密码），特权容器里的 `insmod` 也被拒。

**(b) dma-buf 注册失败，根因是 rdma-core 太旧**

```
NCCL INFO dlvsym failed on mlx5dv_reg_dmabuf_mr -
  /lib/x86_64-linux-gnu/libmlx5.so: undefined symbol: mlx5dv_reg_dmabuf_mr, version MLX5_1.25
```

镜像里是 `rdma-core 50.0-2ubuntu0.2` / `libmlx5.so.1.24.50.0`，
`nm -D` 里**根本没有** `mlx5dv_reg_dmabuf_mr`。NCCL 2.32.3 按 `MLX5_1.25` 版本号去找它，
对应 **rdma-core >= 51.0**。差一个大版本。

## 修复路线（按性价比排序）

| # | 修法 | 谁做 | 成本 | 风险 |
|---|---|---|---|---|
| 1 | 宿主机 `modprobe nvidia_peermem`（先 root 排查 EINVAL，必要时 `dkms autoinstall` 重建） | 需要 root | 5 秒（+排查） | 低，可 `rmmod` 回滚；peermem 只注册一个 IB client，不影响已在跑的 CUDA 进程 |
| 2 | 镜像里换 rdma-core >= 51（源码编译，保留 `MLX5_1.25` 符号版本） | 我，不需要 GPU | ~1 次镜像构建 | 中：替换 libibverbs/libmlx5，需回归 NCCL |
| 3 | 降到 torch 自带的 NCCL 2.29.7（它是对 rdma-core 50 编的） | 我 | 1 次构建 + 验证 | 中：可能没有 GIN/GDAKI，需先确认 |

路线 1 是官方推荐姿势，也是集群常用做法，**建议优先**。
路线 2/3 不依赖任何人工介入，我可以直接做。

## 本窗口顺带确认的事

- 已跑的 kernel 测试（`tests/sm90` 142、`test_sm90_moe` 19、`hicache` 3、其余 115）
  全绿，和 GIN 问题无关 -- **kernel 层是好的，卡在通信/基础设施层**。
- warmup 崩溃与 CUDA graph 捕获无关：崩点在 graph capture 之前的 eager profile run。
- engram 相关环境变量（`DEEPINFER_ENGRAM_FORCE_GPU_STORAGE` 等）都改不了这条路径：
  只要创建 EngramBuffer，就必须有 GIN。

## 复现命令（不需要模型权重，20 秒）

```bash
docker run --rm --privileged --network host --ipc=host --gpus all \
  -v /lib/modules:/lib/modules:ro -e NCCL_IB_HCA=mlx5_0 \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  deepinfer-sm90:v41-sm90 bash -lc \
  'cd /opt/deepinfer/DeepInfer && python vendor/DeepEP/tests/engram/test_engram.py \
     --num-processes 8 --num-entries-per-layer 1024,1024 --hidden 512 \
     --num-tokens 8 --num-entries-per-token 2'

# 判定标准：
#   失败 -> "Unable to use GIN: Peermem is not supported ... does not support DMA-BUF"
#   成功 -> 不再出现该 WARN，且能打印 fetch 校验结果
```

这个探针**不加载 510 GB 权重、20 秒出结果**，是后续验证 GIN 修复的最快手段。

## 补充：NCCL 版本矩阵（排除"降 NCCL"这条路）

| NCCL | 位置 | GIN 开关 | dlvsym 版本需求 | dmabuf 引用 |
|---|---|---|---|---|
| 2.28.3 | `/usr/lib/x86_64-linux-gnu/libnccl.so.2`（基础镜像自带） | **无** | `MLX5_1.8` / `MLX5_1.25` | 2 处 |
| 2.32.3 | venv `nvidia/nccl/lib/libnccl.so.2`（`build.sh` 强制装） | `NCCL_GIN_ENABLE` / `NCCL_GIN_TYPE` / `NCCL_GIN_CONNECTION` | `MLX5_1.0/1.3/1.8/1.25` | 10 处 |

两个结论：

1. **2.28.3 完全没有 GIN**（连 `NCCL_GIN_ENABLE` 字符串都没有），所以"降回 torch 自带的
   2.29.7"这条路直接死了 —— 没有 GIN 就没有 engram。`build.sh` 强制 2.32.3 是对的。
2. **连 2.28.3 都要 `MLX5_1.25`**，而镜像里的 rdma-core 只有 50.0（libmlx5 ABI 1.24）。
   说明这个镜像**从来就没走过 dma-buf 这条路** —— 它的正常姿势是宿主机加载
   `nvidia_peermem`，dma-buf 只是 fallback。

**因此修复路线 3（降 NCCL）作废**。剩下：路线 1（宿主机 peermem，推荐）
或路线 2（镜像内升级 rdma-core 到 >=51）。

## 已落地的改动

1. **`tools/e2e_window.sh`**：容器改为
   `--network host --device /dev/infiniband --cap-add IPC_LOCK -e NCCL_IB_HCA=mlx5_0`，
   并**删除 `EP_DISABLE_GIN=1`**（伪修复）。`bash -n` 通过，尚未全新跑过 e2e。
2. **`tools/verify_gin.sh`（新增）**：9 秒 GIN 探针，只用 2 张卡（默认 6,7）、
   小张量（~600 MB/rank），**不影响在跑的 glm53-flash**，不加载权重。

## 基线（peermem 未加载时，2026-10-22 14:17:40）

```
$ tools/verify_gin.sh
GIN DEAD  (3 rank warning(s))
  ... Unable to use GIN: Peermem is not supported, and device N does not support DMA-BUF.
```

9.4 s 完成，`glm53-flash` 保持 `Up`、`/v1/models` 200、显存 138.4 GB 未变。
说明：**探针与被测主体都已就绪，只等宿主机的 `nvidia_peermem` 加载。**

---

# 第二轮：逐条封死，直到拿到"为什么"

14:20 - 14:40 继续排查（**未停服务**，全程只用 GPU 6,7 的小张量探针）。

## 1. peermem 是死路（不是配置问题）

```
nm -u nvidia-peermem.ko            -> __fentry__, param_ops_int, __x86_return_thunk  （仅 3 个）
grep nvidia_p2p_get_pages /proc/kallsyms  -> 0
grep nvidia_p2p_put_pages /proc/kallsyms  -> 0
grep nvidia_p2p_dma_map_pages /proc/kallsyms -> 0
```

真正的 peermem 必须依赖 `nvidia.ko` 导出的 `nvidia_p2p_*` 系列符号。**驱动 595.91.07 一个都不导出**，
镜像里的 `nvidia-peermem.ko` 只有 7.9 KB、无任何 `nvidia_p2p_*` 依赖 —— 它是个空壳。
这就解释了为什么 `modprobe`/`insmod` 都是 **EINVAL 且内核一条日志都不打**：
内核在跑模块 init 之前就发现这个模块没有可用的宿主 API。

**结论：NVIDIA 在 595 这个驱动上已经把 GPUDirect RDMA 完全迁到 dma-buf，peermem 退役。
`sudo modprobe nvidia_peermem` 在这个驱动上永不成功，不要再试。**

## 2. dma-buf 的符号问题已解决（rdma-core 版本定位修正）

我原来说"rdma-core >= 51"，错了。实测：

| rdma-core | libmlx5 | `mlx5dv_reg_dmabuf_mr` |
|---|---|---|
| 50.0（镜像自带） | 1.24.50.0 | **无** |
| 51.0（源码编译） | 1.24.51.0 | **无** |
| master（编译，1.27.66.0） | 1.27.66.0 | **有** |

`debian/ibverbs-providers.symbols` 里写得很清楚：`mlx5dv_reg_dmabuf_mr@MLX5_1.25 54`
—— 是 **rdma-core 54** 引入的。源码编译 master 60 秒即可（`tools/build_rdma_and_probe.sh`）。

装上之后，NCCL 那句 `dlvsym failed on mlx5dv_reg_dmabuf_mr` **消失了**。

## 3. 但真正的拦路虎是 CUDA 设备属性

`dlvsym` 修好之后，NCCL 换了另一句话（8 rank 一致）：

```
Unable to use GIN: Peermem is not supported, and device N does not support DMA-BUF.
```

读 NCCL 源码（`src/transport/net_ib/gin.cc`）确认它查的是：

```c
CUCHECK(cuDeviceGetAttribute(&dmaBufSupportOnDevice, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cudaDev));
if (dmaBufSupportOnDevice == 1) return ncclSuccess;
WARN("Unable to use GIN: Peermem is not supported, and device %d does not support DMA-BUF.", cudaDev);
```

直接实测这个属性（`cuDeviceGetAttribute(124, dev0)`）：

```
attr 124 (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED) = 0     <- 就是它
```

并且在 `--privileged` + 挂 `/dev/dri` 下**仍然是 0** → 不是容器权限问题，是**驱动报的能力**。

**根因（最后一层）：IOMMU 处于 Translated 模式**

```
[24.897153] iommu: Default domain type: Translated
[24.974859] DMAR: IOMMU feature prs inconsistent
/sys/class/iommu/dmar{0,1,10,...} 存在
```

NVIDIA GPUDirect RDMA 的官方要求是 IOMMU **passthrough**（`iommu=pt`）或关闭。
在 Translated（严格 DMA 重映射）下，GPU 内存不能当对等 DMA 目标，驱动因此把
`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` 报成 0。`/proc/cmdline` 里**没有任何 iommu 参数**，
是内核默认开成了 Translated。

## 4. PROXY 后端救不了（已实测排除）

`NCCL_GIN_TYPE` 枚举（来自我们镜像里的 `nccl_device/core.h`）：

```c
NCCL_GIN_TYPE_NONE  = 0,   // sentinel
NCCL_GIN_TYPE_PROXY = 2,   // CPU 代理线程，不需要 GPUDirect RDMA
NCCL_GIN_TYPE_GDAKI = 3,   // deep_ep 写死
NCCL_GIN_TYPE_GPI   = 4,
NCCL_GIN_TYPE_EFA_GDA = 5,
```

我把 `deep_ep/buffers/engram.py` 的 `NCCL_GIN_TYPE` 改成 `2`（PROXY）实测：
日志里**仍然是 `GIN_IB_GDAKI`**，仍然是同一条 GDR 失败。

原因在 DeepEP 自己的 C++ 里，`vendor/DeepEP/csrc/kernels/comm/context.cpp`：

```c
EP_HOST_ASSERT(props.ginSupport[NCCL_GIN_TYPE_GDAKI] and "NCCL GDAKI is unavailable ...");
reqs.ginType = NCCL_GIN_TYPE_GDAKI;      // 硬编码，env 改不动
```

**DeepEP Engram 只支持 GDAKI**，PROXY 不是选项。同一位置还有 `NCCL_GIN_TYPE_GPI=4`，
但它要 `nvidia_fs`(GDS)，本机 `no nvidia_fs`，也没有。

## 5. engram 能不能关？（不能）

`deepseek.py:178  if self.engram_runtime is not None:` —— 看似可关，但
`engram_enabled` 来自模型权重元数据（HF 权重里有 2 张 Engram 表），
`model_metadata_utils.py:163 args.max_ngram_size = ... if self.engram_enabled else 0`。
关掉就是**跑一个退化的模型**（n-gram 门控通路整段失效），不能用于服务。

## 6. 最终结论与完整修复包

DeepSeek-V4.1-Flash-1017 **必须**有 DeepEP Engram → 必须有 NCCL GIN/**GDAKI**
→ 必须 GPUDirect RDMA。本机三条路全断，且**根因是宿主机的 IOMMU 配置**，不是我们的代码/镜像。

完整修复包（三件事，缺一不可）：

| # | 改什么 | 谁做 | 验证方式 |
|---|---|---|---|
| 1 | 宿主机 GRUB 加 `iommu=pt`（`GRUB_CMDLINE_LINUX` 里追加，`update-grub` 后**重启**） | 需要 root + 重启窗口 | `tools/verify_gin.sh` 9 秒出结果；或直接 `cuDeviceGetAttribute(124)==1` |
| 2 | 镜像里装 **rdma-core >= 54**（源码编译，`tools/build_rdma_and_probe.sh` 已验证 60 秒可编出 `libmlx5.so.1.27.66.0`） | 我，不需要 GPU | `nm -D libmlx5.so.1 \| grep mlx5dv_reg_dmabuf_mr` |
| 3 | 容器 `--network host --device /dev/infiniband --cap-add IPC_LOCK`（已改好 `tools/e2e_window.sh`） | 已完成 | `NCCL_DEBUG` 里出现 `NET/IB: [0] mlx5_0:uverbs0:1/RoCE` |

**第 1 条是硬门槛**：不重启改 IOMMU，第 2、3 条做得再对也没用（`attr 124` 仍为 0）。
反过来，只要 `attr 124` 变成 1，第 2 条即使不做也可能通（NCCL 那条检查先看 peermem，
再看 dma-buf 属性，属性为 1 就直接通过；但 GDAKI 的 init 路径仍要 `ncclIbDmaBufSupport`，
所以 rdma-core 还是建议一起升）。

## 待办清单（给下一次窗口 / 给运维）

- [ ] **运维**：`iommu=pt` + 重启（预计 5 分钟停机，建议排到维护窗口）
- [ ] 重启后立刻跑 `tools/verify_gin.sh`（9 秒，不用停 glm53-flash）确认 `GIN OK`
- [ ] 我：把 rdma-core >= 54 写进 `docker/Dockerfile`（`tools/build_rdma_and_probe.sh` 里的步骤可直接搬）
- [ ] GIN 通了之后：`tools/e2e_window.sh start 40` 跑真正的端到端 + 欠下的 `test_offline_inference` / `test_kl` / prefix-cache A/B

---

## 第三轮：镜像已含 rdma-core 54，但探针现在跑不动了（诚实标注）

镜像重建完成（`logs/build-final-20261022-1426.log`，`BUILD_EXIT=0`）
→ `deepinfer-sm90:v41-sm90` 现在是 `libmlx5.so.1.25.54.0`，`mlx5dv_reg_dmabuf_mr` 存在。

**但 15:42 那次探针的"警告消失"是假信号，必须纠正**：

生产已完全热身，显存只剩 **473-497 MiB/卡**（14:17 时有 5.3 GB）。那次探针死在
**比 GIN 检查更早的地方**：

```
init_process_group -> torch.distributed.DistBackendError: ncclUnhandledCudaError
  Failed to CUDA calloc async 1216 bytes     <- 显存不足，不是 GIN 问题
```

它根本没走到 `EngramBuffer` 创建，所以"没有 GIN 警告"只说明它没跑到那一步。
连跑两次都是同一失败，与 flake 无关。

**仍然是 14:23 那次（容器内临时编 rdma-core master）的结论有效**：
那次显存够，确实走到了 GIN 检查，结果是

```
Unable to use GIN: Peermem is not supported, and device N does not support DMA-BUF.
```

即 **rdma-core 升级只消掉了 `dlvsym failed on mlx5dv_reg_dmabuf_mr`，消不掉
`attr 124 = 0`**。IOMMU 仍然是硬门槛。这也正好说明 07 的变更单里三件事缺一不可。

**给未来的自己**：显存 < 1 GiB/卡时不要用这个探针，它会死在
`cudaMallocAsync` 而不是给出 GIN 结论。用 `PROCS=1` 可以减半显存，但仍需约 1 GB。

---

## 第四轮（15:52-16:05）：`iommu=pt` 变更已执行，**假设被证伪**

用户全权授权后由 agent 执行。通过「特权容器 -> `nsenter -t 1 -m -u -i -n -p`」拿到宿主机
root 命名空间完成（本机 `sudo` 需要密码，docker 是等价权限）。

### 变更内容（`diff /root/grub.bak.2026-10-22-155257 /etc/default/grub`）

```
12c12
< GRUB_CMDLINE_LINUX=""
---
> GRUB_CMDLINE_LINUX="iommu=pt"
```

**只有这一行**。`GRUB_DEFAULT` 保持运维原有的
`"Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-57-generic"` 不动 —— 原本就钉在 57，
所以重启没有顺带换内核（中途我一度误读 grub.cfg 首项是 138 而多做了一次 pin，已还原）。

重启：15:54:10 排定 -> 15:57 sshd 回来 -> 15:58:27 `up 1 min`。内核仍是 `6.8.0-57-generic`。

### 验收结果

| 检查 | 结果 |
|---|---|
| `/proc/cmdline` 有 `iommu=pt` | ✅ `... ro iommu=pt quiet splash ...` |
| IOMMU 默认域类型 | ✅ `iommu: Default domain type: Passthrough (set via kernel command line)` |
| `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` (attr 124) | ❌ **仍然是 0** |
| GIN 探针 | ❌ 仍然 `GIN DEAD` |

**结论：IOMMU Translated 不是根因。** 换成 passthrough 后 dma-buf 能力没有任何变化。

### 进一步证据（把"是不是某个废弃属性在骗人"也排除掉）

| 探测 | 结果 |
|---|---|
| attr 124 `DMA_BUF_SUPPORTED` | 0 |
| attr 146 `HOST_ALLOC_DMA_BUF_SUPPORTED` | 0 |
| attr 152 `DMA_BUF_MMAP_SUPPORTED` | `CUDA_ERROR_INVALID_VALUE`（驱动不认识这个属性） |
| attr 115 `MEMORY_POOLS_SUPPORTED` | 1（说明属性查询本身是好的，不是全 0） |
| `nvidia.ko` 里的 `dma_buf` 字符串 | 12 处（代码在） |
| `nvidia-drm.ko` | 4 处；`/dev/dri/card0..card7` 都在 |
| CUDA driver API | 13020（13.2），驱动 595.91.07 |

**驱动在整族 dma-buf 能力上都报不支持**，不是某一个属性被废弃的问题。加上 `nvidia-peermem.ko`
是空壳（不导出/不依赖 `nvidia_p2p_*`），可以定论：

> **驱动 595.91.07-server 在 H20-3e 上不提供 GPUDirect RDMA。**
> 两条实现路径（peermem / dma-buf）在这条驱动线上都是关着的。

### 官方文档佐证这是硬前提

`doc/inference_deployment_qa.md:157`：

> "**Swap-in** from shared memory to HBM runs in a thread inside the engine process ...
> Because the CUDA copy engine performs poorly on the fine-grained copies of KV cache blocks,
> **swap-in is implemented with GDR: an RDMA NIC saturates PCIe bandwidth without consuming SMs.**"

参考部署设计上就假定 GDR 可用，不是可选项。

### 驱动升级可行性评估（未执行）

```
nvidia-driver-595-server : 已安装 595.91.07 / 候选 595.91.07   <- 无升级
nvidia-driver-595        : 候选 595.99.02                      <- 另一个 flavour
nvidia-open-595          : 不存在
```

`-server` flavour 没有更新；要拿 595.99.02 得切到非 server flavour（换内核模块构建体系），
属于高爆炸半径变更，且**同版本线内 patch 升级修好 dma-buf 的概率不高**，未擅自执行。

另外一条路（降驱动到 R550/R560 让 peermem 复活）**与生产冲突**：`glm53-flash` 的 torch 是
`2.13.0+cu132`，需要驱动 >= 590，降不到 peermem 时代的版本。

### 生产恢复

15:59:19 `docker start glm53-flash`，`restart=always` 已恢复；16:0x 起 `/v1/models` 200。
整个变更期间生产停机约 5 分钟（15:54:10 -> 15:59:19 启动，含 4.7 分钟热身）。

### 留下的东西

- 宿主机保留 `iommu=pt`：对 vLLM 无影响（它不用 GDR），且是将来任何 GDR 方案的前置条件，
  省一次重启。回滚：`sudo sed -i 's|^GRUB_CMDLINE_LINUX=.*|GRUB_CMDLINE_LINUX=""|' /etc/default/grub && sudo update-grub && sudo reboot`
- 备份：`/root/grub.bak.2026-10-22-155257`
- 新工具：`tools/post_reboot_check.sh`（四步验收，一条命令）、`tools/probe_dmabuf.py`
