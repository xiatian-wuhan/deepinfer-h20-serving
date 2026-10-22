# 23 — 换开源内核模块，GIN 修好了

日期：2026-10-22。执行人：自动化。回滚材料在 `/root/driver-swap/`。

## 做了什么

专有内核模块 → 开源（`nvidia-driver-595-server-open`），**同版本 595.91.07，不是升级**。

关键认知：open 与专有只差**内核模块**，userspace 驱动包是同一份。所以只换内核侧三个包：

```bash
apt-mark unhold $(apt-mark showhold | grep -i nvidia)      # 原本被 hold 住，会挡住替换
apt-get install -y nvidia-kernel-source-595-server-open nvidia-dkms-595-server-open
# 事务：移除 nvidia-{driver,dkms,kernel-source}-595-server，装入对应 -open
```

没有用 `nvidia-driver-595-server-open` 元包，因为它会拖进一堆 **i386 桌面库**
（`libnvidia-gl-595-server:i386`、`nvidia-settings`、`nvidia-prime`…），对无头计算节点是污染。

## 验证结果

| 项 | 切换前 | 切换后 |
|---|---|---|
| `NVRM ... ` | `Kernel Module` | **`Open Kernel Module`** |
| `modinfo -F license nvidia` | `NVIDIA` | **`Dual MIT/GPL`** |
| `nvidia-peermem.ko` | 桩，`init_module` 恒返回 `-EINVAL` | `Dual BSD/GPL`（开源实现） |
| **`cuDeviceGetAttribute(124, DMA_BUF_SUPPORTED)`** | **0** | **1** |
| dma-buf 导出 | OK（fd 可拿） | OK |
| **`deepinfer` GIN 探针** | **GIN DEAD** | **`GIN OK (exit 0, IB up, no GDR warning)`** |
| glm53-flash (vLLM) | — | 正常：模型列表 + 生成（`1+1 → 2`），usage 正常 |
| 本服务 25 项协议回归 | 25/25 | **25/25** |

`ibv_reg_dmabuf_mr` 单独调用在本机仍是 `EINVAL`（可能受 `iommu=pt` 影响），但这**不影响 GIN**——
NCCL 的 GDAKI 走的是自己的注册路径，探针实测已通过。

## 性能：GIN 比旧的 NVLink 回退快

同一把尺子（`tools/serve_bench.py`，1043 token prompt，c=1，热态）：

| Engram 路径 | TPOT p50 | 单流 |
|---|---:|---:|
| `EP_DISABLE_GIN=1` + PR#99 NVLink（旧回退） | 6.5 ms | 153.0 tok/s |
| **GIN 原生（默认）** | **5.6 ms** | **177.9 tok/s** |

GIN 快约 **14%**，并且不再需要 PR#99 的 NVLink 缓冲，显存少占约 **6 GB/卡**
（122 GB vs 128 GB）。

## 终态配置

```
CUDA_VISIBLE_DEVICES=0..7 DEEPINFER_HF_MODEL_DIR=/model \
DEEPINFER_USE_NEXTN_DECODE=5 DEEPINFER_ENABLE_MTP_VARLEN=1 \
serve/server.sh start --dp 8 --cp 8 --max-num-seqs 32 --port 8888 --log-requests
```

**不再设 `EP_DISABLE_GIN=1`**（那会让 DeepEP 回退到 NCCL 并启用 NVLink Engram 回退路径）。
`DEEPINFER_ENGRAM_NVLINK_FETCH` 的默认值跟着 `EP_DISABLE_GIN` 走，不设即走 GIN 原生路径。

## 运维注意

- `pkill -f "xxx"` 会自杀（pkill 自身命令行也匹配），要用 `pkill -f 'xx[x]'` 形式。
- 通过 `docker run --privileged --pid=host ... nsenter` 执行 `systemctl reboot` 时**必须前台**，
  放后台会随容器 cgroup 一起被杀（实测踩过一次，`uptime` 没变才发现）。
- 重启后 SSH 握手会短暂失败（"Connection lost before handshake"），等 1-2 分钟即可。

## 回滚

```bash
apt-get install -y nvidia-driver-595-server nvidia-dkms-595-server nvidia-kernel-source-595-server
# 会移除 -open 三件套，装回专有；然后 reboot
```
原始包列表 / hold 列表 / grub / modprobe.d 备份在 `/root/driver-swap/`。
