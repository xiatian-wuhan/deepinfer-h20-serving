# 变更单：h20 宿主机 IOMMU 改为 passthrough（`iommu=pt`）

主机 `<h20-host>`（k8s-node-238）· H20-3e ×8 · 驱动 595.91.07 · 内核 6.8.0-57-generic
提出：2026-10-22 15:45 · 类型：**需要重启** · 预计停机 **8-12 分钟**

---

## 1. 为什么必须改

要让 DeepSeek-V4.1-Flash-1017 在这台机器上跑起来，绕不过 DeepEP 的 **Engram**
（模型的 n-gram 嵌入表，HF 权重里有 2 张表）。Engram 的取表内核走
**NCCL GIN / GDAKI**，而 GDAKI 必须有 **GPUDirect RDMA**。

本机 GPUDirect RDMA 的两条路都断了，且**根因在宿主机的 IOMMU 配置**：

```
CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED (attr 124) = 0      # 驱动报的能力
iommu: Default domain type: Translated                     # 内核日志
```

NCCL 源码 `src/transport/net_ib/gin.cc` 里，GIN 的可用性判定就是查这个属性：

```c
CUCHECK(cuDeviceGetAttribute(&dmaBufSupportOnDevice, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cudaDev));
if (dmaBufSupportOnDevice == 1) return ncclSuccess;
WARN("Unable to use GIN: Peermem is not supported, and device %d does not support DMA-BUF.", cudaDev);
```

而 NVIDIA 对 GPUDirect RDMA 的官方要求是 IOMMU **passthrough** 或关闭。
`/proc/cmdline` 里目前没有任何 iommu 参数，内核默认开成了 Translated（严格 DMA 重映射），
严格模式下 GPU 显存不能作为对等 DMA 目标，驱动因此把该属性报 0。

**另一条路 `nvidia_peermem` 已经确认是死路，不必再试**：
驱动 595.91.07 **不导出任何 `nvidia_p2p_*` 符号**，`nvidia-peermem.ko` 只剩 7.9 KB 的空壳，
`modprobe` 必然 `Invalid argument` 且内核一条日志都不打。NVIDIA 已在此驱动线迁到 dma-buf。
所以**只剩 `iommu=pt` 这一条**。

## 2. 影响面

| 项 | 影响 |
|---|---|
| 重启 | 是。改内核命令行必须重启 |
| glm53-flash（vLLM 生产） | 会随重启中断；容器 `restart=always`，**重启后会自动拉起**，热身约 4.7 分钟 |
| 其他业务 | 本机只跑 glm53-flash 一个容器（另有 docker0/bridge，无别的服务） |
| IOMMU 安全属性 | `iommu=pt` 让 DMA 走 passthrough，绕过 IOMMU 重映射。HPC/GPU 集群的标准配置；代价是失去对恶意设备 DMA 的隔离能力 |
| 性能 | GPUDirect RDMA 打开后，RDMA 可直接读写显存，省掉一次显存↔主存拷贝（是收益，不是风险） |

## 3. 变更步骤

### 3.1 变更前（记录现状，便于回滚）

```bash
cp /etc/default/grub /root/grub.bak.$(date +%F-%H%M)
cat /proc/cmdline                                  # 期望：... ro quiet splash vt.handoff=7
grep -E '^GRUB_CMDLINE' /etc/default/grub          # 期望：DEFAULT="quiet splash"; LINUX=""
docker inspect glm53-flash --format '{{.HostConfig.RestartPolicy.Name}}'   # 期望：always
```

### 3.2 改内核命令行

当前 `/etc/default/grub`：

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX=""
```

改成（只动 `GRUB_CMDLINE_LINUX`，保留 `quiet splash`）：

```
GRUB_CMDLINE_LINUX="iommu=pt"
```

命令：

```bash
sed -i 's|^GRUB_CMDLINE_LINUX=.*|GRUB_CMDLINE_LINUX="iommu=pt"|' /etc/default/grub
grep -E '^GRUB_CMDLINE' /etc/default/grub          # 确认 iommu=pt 已写入
update-grub
```

> 不要同时加 `intel_iommu=on`，本机 DMAR 已经在工作（`/sys/class/iommu/dmar*` 存在），
> 加了反而可能因为 `DMAR: IOMMU feature prs inconsistent` 触发新的问题。

### 3.3 暂时关掉生产的自动重启（关键，否则验证抢不到卡）

`glm53-flash` 是 `restart=always`，重启后会自动把 8 张卡吃满（142.7 GB/卡，只剩 ~0.5 GB），
验证探针跑不起来。

```bash
docker update --restart=no glm53-flash
```

### 3.4 重启

```bash
reboot
```

### 3.5 重启后验收（按顺序，共约 2 分钟）

**Step 1 — 内核命令行生效了没**（10 秒，不需要 GPU 空闲）

```bash
cat /proc/cmdline | tr ' ' '\n' | grep -x 'iommu=pt'        # 期望输出：iommu=pt
dmesg | grep -i 'Default domain type'                        # 期望不再出现 Translated
```

**Step 2 — 驱动现在报不报 dma-buf 支持**（2 秒，显存占用可忽略）

```bash
docker run --rm --gpus '"device=0"' -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  deepinfer-sm90:v41-sm90 python -c "
import ctypes
l=ctypes.CDLL('libcuda.so.1'); l.cuInit(0)
d=ctypes.c_int(); l.cuDeviceGet(ctypes.byref(d),0)
v=ctypes.c_int(); l.cuDeviceGetAttribute(ctypes.byref(v),124,d)
print('attr 124 (DMA_BUF_SUPPORTED) =', v.value)
"
```

- 期望 **`= 1`** → 继续 Step 3
- 仍为 `= 0` → 见第 6 节，**不要**急着恢复生产，先报我

**Step 3 — GIN 到底通没通**（9 秒，需要 2 张空闲卡）

```bash
cd /home/ubuntu/data1/xia/deepinfer-h20
bash tools/verify_gin.sh
```

- 期望 **`GIN OK`**
- 若报 `GIN DEAD ... does not support DMA-BUF` → Step 2 应该是 0，属同一问题
- 若报 `IB NOT VISIBLE` → 容器参数问题，找我

**Step 4 — 恢复生产**

```bash
docker update --restart=always glm53-flash
docker start glm53-flash
```

等待约 4.7 分钟（`init engine 140 s` + `Graph capturing finished 74 s`），然后：

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18881/v1/models   # 期望 200
docker inspect glm53-flash --format '{{.HostConfig.RestartPolicy.Name}}'     # 期望 always
```

> **务必确认这两条都过了再结束变更单**。`restart=always` 没恢复是这次变更最容易留下的坑。

## 4. 回滚

任一验收步骤失败：

```bash
cp /root/grub.bak.* /etc/default/grub     # 或 sed 回空字符串
update-grub
reboot
```

`grub.bak.*` 是 3.1 里备份的原始文件，恢复后与本变更前完全一致。
`glm53-flash` 的 `restart` 策略若已被改成 `no`，记得 `docker update --restart=always glm53-flash`。

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 重启后起不来 / 进不了系统 | 低 | GRUB 在启动菜单里按 `e` 可临时删除 `iommu=pt` 启动；或 rescue 模式恢复备份 |
| `iommu=pt` 加了但 attr 124 仍是 0 | 中 | 说明不是 IOMMU 单方面原因，进第 6 节。**此时生产照常恢复运行**（`iommu=pt` 不影响 vLLM） |
| 忘记恢复 `restart=always` | 中 | Step 4 已把这条写成显式验收项 |
| 重启期间有业务请求 | 确定 | 排维护窗口，提前通知 |

## 6. 如果 `iommu=pt` 没解决问题

按顺序再试（每条都要重新验证 Step 2）：

1. **BIOS**：确认 VT-d / DMA Remapping 已开，且 PCIe ACS 未被过度收紧
   （`DMAR: IOMMU feature prs inconsistent` 提示 DMAR 表有不一致，可能与 BIOS 版本相关）。
2. **降驱动**：换到仍导出 `nvidia_p2p_*` 的驱动线（R550/R560 系列），
   这样 `nvidia_peermem` 能加载，走传统 GDR 路径 —— 但需要确认降驱动不影响 glm53-flash
   所依赖的 CUDA 13.2 特性。
3. **升级内核/驱动到 NVIDIA 认证组合**，并向 NVIDIA 提交
   "H20 + 595.91.07 报 `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0`" 的工单。
4. **换部署形态**：multi-node，让 Engram 走真正的跨节点 RDMA（GIN 在 multi-node 下是设计场景）。

## 7. 变更后的产出（我这边已就绪）

- 镜像 `deepinfer-sm90:v41-sm90` 已含 **rdma-core 54**（`libmlx5.so.1.25.54.0`，
  提供 NCCL dma-buf 路径需要的 `mlx5dv_reg_dmabuf_mr@MLX5_1.25`）。构建日志
  `logs/build-final-20261022-1426.log`，`BUILD_EXIT=0`，镜像自检 7 项通过。
- 容器参数已改为 `--network host --device /dev/infiniband --cap-add IPC_LOCK`（`tools/e2e_window.sh`）。
- 验证工具 `tools/verify_gin.sh`（9 秒；`PROCS=1` 可把显存需求减半）。
