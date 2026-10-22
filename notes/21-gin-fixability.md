# 21 — GIN 不可用到底能不能修（调研）

日期：2026-10-22。结论：**能修，但必须换内核模块形态**（专有 → 开源），不是配置问题。

## 三层根因（每层都有实证）

### 第 1 层：`nvidia-peermem` 是设计上的桩，不是坏了

```
$ insmod nvidia-peermem.ko      -> -EINVAL，内核零日志
$ objdump -d nvidia-peermem.ko  -> init_module:
      mov $0xffffffea, %eax     ; -22 = -EINVAL
      jmp __x86_return_thunk    ; 直接返回，什么都没做
```

模块里保留了 `peerdirect_support` / `persistent_api_support` 参数和 `nv_mem_client_init`
符号，但入口函数恒返回 EINVAL。原因是它依赖 Mellanox 专有的
`ib_register_peer_memory_client()`，该符号**只存在于 MLNX_OFED 编的 ib_core**，
Ubuntu 的 inbox rdma-core 里没有。所以在这套软件栈上 peermem 永远不可能加载。
（旁证：`nvidia.ko` 本身其实导出了 `nvidia_p2p_get_pages/put_pages`，
kallsyms 里可见 12 处；但客户端模块被桩掉了。）

### 第 2 层：NCCL 的 GIN 关卡在 `ncclGinConnectOnce`

```
NCCL INFO GIN/Plugin: Could not find: libnccl-gin.so
NCCL INFO gin/gin_host.cc:198 (ncclGinConnectOnce) -> 5
RuntimeError: ... 5 (Unable to use GIN: Peermem is not supported, and device 0 does not support DMA-BUF.)
```

这句话在 `libnccl.so.2`（2.32.3）里出现 2 次，所以判定在 NCCL 内部，不是 DeepEP。
加 `NCCL_DMABUF_ENABLE=1`、`NCCL_GIN_ENABLE=1` 重试，报错完全不变。

NCCL 官方文档（2.29）的 GIN 前置条件：
> GIN host proxy requires **DMA-BUF or nvidia-peermem**. GIN GDAKI requires
> **DMA-BUF with kernel version >= 6.1** or nvidia-peermem。

即：要么 peermem（第 1 层已死），要么 DMA-BUF。

### 第 3 层：DMA-BUF 能用一半 —— 导出成功，但 RDMA 注册失败

内核侧前置条件**全部满足**：

| 项 | 实测 |
|---|---|
| `CONFIG_DMA_SHARED_BUFFER` / `CONFIG_HMM_MIRROR` / `CONFIG_INFINIBAND_ON_DEMAND_PAGING` / `CONFIG_DMABUF_MOVE_NOTIFY` | 全 `=y` |
| `ib_umem_dmabuf_*` 符号 | kallsyms 里 29 处 |
| mlx5_0 ODP | `ODP_SUPPORT` + `ODP_SUPPORT_IMPLICIT` |
| `nvidia-drm modeset` | `Y` |
| 内核 / rdma-core | 6.8.0-57 / 54（文档要求 ≥6.1 / ≥44） |

三连测结果：

| 调用 | 结果 |
|---|---|
| `cuMemCreate` + `cuMemExportToShareableHandle(POSIX_FD)` | **OK**，fd=66 |
| `cuMemMap` + `cuMemSetAccess` | OK，映射成功 |
| `ibv_reg_dmabuf_mr(pd, 0, 2MiB, ptr, fd, ...)` | **FAIL，errno=22 EINVAL** |
| `cuDeviceGetAttribute(124, DMA_BUF_SUPPORTED)` | **0**（`attr 146` 也是 0，`115`=1） |

先前一度以为"attr 124 是假象、导出成功就说明能用"。**这个判断现在被推翻**：
导出 fd 成功只说明图形栈路径可用，而 RDMA 注册被内核静默拒绝（dmesg 无任何记录）。
**attr 124 = 0 是准确的**：这台机器上的 dma-buf 不是 RDMA 可注册的。

根因是内核模块形态：`modinfo -F license nvidia` = `NVIDIA`（**专有模块**）。
NVIDIA 的 DMA-BUF GPUDirect RDMA 路径要求**开源内核模块**（nvidia-open），
专有模块的 dma-buf 导出不带 RDMA 所需的 ops。

## 修复选项

| 方案 | 可行性 | 代价 | 风险 |
|---|---|---|---|
| **换 `nvidia-driver-595-open`** | apt 有 595.99.02，Hopper 支持 | 装包 + 重启 | 驱动版本 595.91.07→595.99.02；**生产 vLLM 服务和本服务都要重新验收**；开源模块性能在数据中心卡上通常等价，但未实测 |
| 装 MLNX_OFED 让 peermem 可用 | OFED 里有 `ib_register_peer_memory_client` | 替换整套 IB 用户态栈 + 重启 | OFED 对 595 驱动的支持不确定；会动到整机网络栈，影响面最大 |
| 拦截 dlsym 伪造 attr 124 | 理论可行 | 需在镜像里做 libcuda 垫片 | NCCL 是 `dlopen("libcuda.so.1")`+dlsym（不 link libcuda），LD_PRELOAD 挡不住，得拦 dlsym 或伪造 libcuda；而且**即使骗过 NCCL，第三层注册仍会 EINVAL**，白费劲 |
| 维持现状（`EP_DISABLE_GIN=1`） | — | 0 | Engram 走 PR#99 的 NVLink 路径，已优化到参考的 96% |

**结论**：只有第一条路是真的能通。第三条已被第三层实验否掉——问题不在 NCCL 的判断，
而在驱动确实提供不了 RDMA 可注册的 dma-buf。

## 值不值得换

GIN 能带来的收益（当前差距）：

- EP dispatch/combine 走 GIN 而非 NCCL 回退：预填 profile 里这两项占 21%；
- Engram 可取原生路径，不必依赖 PR#99 的 NVLink 实现；
- 可能缩小 CP8 128K 预填 3.064 vs 参考 2.114（差 45%）、c=32 吞吐 1777 vs 2912 的差距。

但代价是整机驱动更换 + 重启 + 两套服务重新验收。当前 Engram 修复后 DP8 c=1 已达参考的 96%，
GIN 的边际收益主要在**预填**和**高并发**。

建议：等有维护窗口时再做，并且先在窗口内做单点验证——
换模块后立刻跑 `tools/verify_gin.sh`，通过再谈收益；不通过就回滚。

## 复现命令

```bash
# 取证：模块是桩
objdump -d /lib/modules/$(uname -r)/updates/dkms/nvidia-peermem.ko | sed -n '/<init_module>:/,+6p'
# 三连测（容器内）
docker cp /tmp/dmabuf_gdr.c di-e2e:/tmp/ && docker exec di-e2e bash -lc \
  'cd /tmp && gcc -o dmabuf_gdr dmabuf_gdr.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -libverbs && ./dmabuf_gdr'
# GIN 探针
tools/verify_gin.sh
```

---

## 结论更新（2026-10-22 晚）：已修复

上面推断的"换开源内核模块"这条路**当天就验证通过了**，详见 `notes/23-driver-swap-gin.md`。

- 安装 `nvidia-driver-595-server-open`（同版本 595.91.07，只换内核模块，userspace 不动）→ 重启；
- `cuDeviceGetAttribute(124, DMA_BUF_SUPPORTED)` 由 **0 → 1**；
- GIN 探针由 **GIN DEAD → GIN OK**；
- 两套服务（glm53 vLLM + 本服务）均重新验收通过，本服务 25/25；
- 且 GIN 原生路径比原来的 NVLink 回退**快约 14%**（TPOT 5.6 vs 6.5 ms）。

所以"GIN 不可用"这条已从限制列表移除。
