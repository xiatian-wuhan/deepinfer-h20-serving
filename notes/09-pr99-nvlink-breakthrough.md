# 转折点：上游 PR #99 提供了不依赖 GIN 的 Engram 取表路径

发现时间：2026-10-22 16:08-16:12（**全程未停 glm53-flash**）
来源：GitHub 检索（用户指示"先去 github 上找一下"）

---

## 一句话

我们不是撞上死路，而是撞上一个**已知且已有解**的问题：
[`deepseek-ai/DeepInfer#99`](https://github.com/deepseek-ai/DeepInfer/pull/99)
*"Add an NVLink Engram fetch for single-node runs without NCCL GIN"*
为单节点、无 GIN 的机器加了一条**纯 NVLink 的 Engram 取表**，**不需要 GPUDirect RDMA**。

## 检索路径

`gh-proxy` 的 GitHub API 配额被用光了（5000/5000），但 **`api.github.com` 从本机可直连**，
改用直连后一路查通：

1. [DeepEP#789](https://github.com/deepseek-ai/DeepEP/issues/789)
   *"EngramBuffer: single-node NVLink support without RDMA devices?"*
   —— 创建于 2026-10-20，环境与**我们几乎一模一样**：驱动 `595.91.07`、
   PyTorch `2.13.0+cu132`、NCCL `2.32.3`、DeepEP `93eb6eb`、单节点 NV18。
2. 该 issue 的唯一回复（kbespalov）给了第二个数据点：**8×B200 + 真 IB ConnectX-7 全 Active，
   仍然失败**在 GDAKI 上下文创建
   （`GIN: DevComm setup failed on all available backends`）；
   加 `EP_DISABLE_GIN=1` 后"首次前向 `cudaErrorIllegalAddress`"——与我们的症状逐字相同。
   并指出："同一个 DeepInfer 构建前一天在同云 8×B300 上跑得好好的，
   所以取决于宿主 RDMA 是否支持 GDAKI"，并指向 DeepInfer#99。
3. [DeepEP#629](https://github.com/deepseek-ai/DeepEP/issues/629)
   *"EP_DISABLE_GIN=1 does not bypass NCCL Gin preflight check"*
   → [DeepEP#630](https://github.com/deepseek-ai/DeepEP/pull/630)（**已合并**，+11/-8）
   但它只改 `csrc/kernels/backend/nccl.cu`（**EP 路径**），让 `EP_DISABLE_GIN` 跳过断言；
   **不涉及 Engram 的 `csrc/kernels/comm/context.cpp`**，所以救不了我们。
4. **DeepInfer#99**（open，5 个文件 +179 行）—— 这才是解。

## PR #99 说的是什么

> `deep_ep.EngramBuffer.fetch` is implemented only on top of GIN. `engram_fetch.cuh` issues
> `comm::NCCLGin` gets **even when `num_rdma_ranks == 1`**, because the shards live in host memory
> owned by other ranks. Without GIN:
> - by default, `EngramBuffer` creation fails with `GIN: DevComm setup failed on all available backends`;
> - with `EP_DISABLE_GIN=1`, the first forward (the profile run) hits `CUDA_ERROR_ILLEGAL_ADDRESS`
>   in the Engram fetch kernel.

这两条**正是我们记录过的两个现象**（blocking 之前的 assert，和 blocking 之后的 ILLEGAL_ADDRESS）。

它的做法（新增 `deepinfer/model_executor/parallel_utils/engram_nvlink.py`，108 行）：

1. **write**：每个 rank 把自己的表分片留在 pinned host memory，用 UVA 零拷贝读；
   FP8 scale 表整份放 GPU（V4.1 约 6 GB/rank）。
2. **fetch**：indices 补齐到 `max_num_tokens` 后 `all_gather`；
   每个 rank `index_select` 自己拥有的行、其余补零；一次 `all_to_all_single` 收齐。
   全程只用 NCCL 集合通信（走 NVLink），**不碰 GIN**。
3. 开关：`DEEPINFER_ENGRAM_NVLINK_FETCH`，未设置时**跟随 `EP_DISABLE_GIN`**。

## 已完成的验证（生产在跑的情况下）

补丁**干净应用**到我们的树（`885eea2` + SM90/144/195/147），仅 6 行 / 1 行偏移；
已提交为新分支 `pr99-engram-nvlink`（`63b430a`），补丁留档 `patches/pr99-engram-nvlink.patch`。

| 验证 | 结果 |
|---|---|
| `git apply --check` | ✅ 干净 |
| PR 自带的 `tests/distributed/test_engram_nvlink.py`（2 卡，比对 full-table 直查结果 + TMA 对齐 stride） | ✅ **1 passed in 4.98s**，**在 glm53-flash 运行中完成** |
| `EP_DISABLE_GIN=0` → 开关 | ✅ `False`（默认行为不变） |
| `EP_DISABLE_GIN=1` → 开关 | ✅ `True`（自动启用） |
| 显式 `DEEPINFER_ENGRAM_NVLINK_FETCH=0` | ✅ `False`（可覆盖） |

单测覆盖得相当实在：两个 rank、参差 token 数（`MAX=64 / 1 / 7+rank / 33-rank`）、
逐元素比对取回的 FP8 行与 scale、并断言 `fetched_sf.stride() == (1, ceil(T/4)*4)`。

## 还没验证的（需要窗口）

- **整模型端到端**：`EngramRuntime.initialize()` 传 CPU 分片、
  各层消费 fetch 返回的 hooks——这条集成路径单测没覆盖。
  需要 8 卡 + 510 GB 权重，约 15-20 分钟窗口。
- 显存预算变化：NVLink 模式把 FP8 scale 表整份放 GPU（约 6 GB/rank），
  且表分片留在 pinned host memory，需要确认不会顶到 `gpu_memory_utilization`。

## 对之前工作结论的影响

- **不再需要 GPUDirect RDMA**：`iommu=pt`、rdma-core 54、`nvidia_peermem` 这一整条线
  对本路径**都不需要**了。宿主机上保留的 `iommu=pt` 是无害的标准配置（对 vLLM 无影响），
  不必为它单独再排一次重启；下次自然维护窗可以一并决定去留。
- `notes/08-nvidia-escalation.md` 仍然有效，但**优先级下降**：
  它只在"要走真 RDMA（多节点/跨机 Engram）"时才需要推进。
- 镜像里已有的 rdma-core 54 保留（无害，且多节点时用得上）。

## 下一步

1. 排一个约 20 分钟的窗口（`tools/e2e_window.sh start 20`），
   用 bind-mount 方式挂上 5 个改动文件（**不用重建镜像**），
   带 `EP_DISABLE_GIN=1` 跑 `examples/offline_inference.py --ep 8`。
2. 通过后把 PR#99 打进 `docker/Dockerfile`（或在 `build.sh` 之后 `git apply`），重建镜像。
3. 然后补完之前欠的：`test_offline_inference`、`test_kl`、prefix-cache A/B、OpenAI/Anthropic 层验证。

---

## 端到端验证：**通过**（2026-10-22 16:10:45 - 16:24:45 窗口）

启动方式（**不重建镜像**，用 bind-mount 挂 5 个改动文件）：

```bash
R=$PWD/repos/deepinfer-sm90-92-p195-p147
bash tools/e2e_window.sh start 20 \
  -v "$R/deepinfer/environ.py:/opt/deepinfer/DeepInfer/deepinfer/environ.py:ro" \
  -v "$R/deepinfer/model_executor/models/v4_modules/engram.py:.../engram.py:ro" \
  -v "$R/deepinfer/model_executor/parallel_utils/userbuffer.py:.../userbuffer.py:ro" \
  -v "$R/deepinfer/model_executor/parallel_utils/engram_nvlink.py:.../engram_nvlink.py:ro"

docker exec -d di-e2e bash -lc 'cd /opt/deepinfer/DeepInfer &&
  EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  python -u examples/offline_inference.py --model_name /model --ep 8 > /e2e-logs/run_pr99.log 2>&1'
```

### 结果

| 阶段 | 结果 |
|---|---|
| Engram 初始化 | ✅ `Created NVLink EngramBuffer (2 layers, Engram 6/8)` —— **走的是 NVLink 路径**，不是 DeepEP GIN |
| 权重加载 | ✅ `HF load phase experts 52.4s, allocated=57.57 GiB` |
| **profile run（以前的崩溃点）** | ✅ **完整跑完**：`Memory profiling results: duration=139.98s, peak_torch_memory=100.92GiB, kv_cache_size=11.06GiB` |
| 生成 | ✅ `step 971` 时仍在正常 decode，`avg_SPS 17.778`, `device_forward 49.9ms` |
| 错误 | ✅ **0** —— 无 `Traceback` / `ILLEGAL_ADDRESS` / `RuntimeError` |

**这是 DeepSeek-V4.1-Flash 第一次在这台机器上真正跑起来。**
以前的 `CUDA_ERROR_ILLEGAL_ADDRESS`（sticky error 最终暴露在 `hc.py:129`）彻底消失。

### 为什么没等到最终输出

`OFFLINE_OUTPUT_LEN = 4096` × 多样本，实测约 2.3 steps/s，需要 20 分钟以上生成，
超出本次 20 分钟窗口。日志已 bind-mount 到宿主机（`logs/e2e/run_pr99.log`），
容器被看门狗删除后仍在。**数值正确性（输出文本/PPL）留待下一个窗口验证。**

### 显存预算（重要）

```
total_gpu_memory=139.80 GiB     initial_memory_usage=64.14 GiB
peak_torch_memory=100.92 GiB    kv_cache_size=11.06 GiB
gpu_memory_utilization=0.85
```

`initial_memory_usage` 里含 NVLink 模式整份放在 GPU 的 FP8 scale 表（约 6 GB/rank）。
`kv_cache_size` 只有 11.06 GiB —— 比预期小，因为 `max_model_len=32768` 且
Engram 的 GPU 侧占用吃掉了余量。**后续要跑长上下文服务，需要重新核算
`gpu_memory_utilization` 与 Engram 显存占比。**

## 下一步

1. ~~bind-mount 验证~~ ✅ 已完成
2. 把 PR#99 打进镜像（树里已提交在 `pr99-engram-nvlink` 分支，Dockerfile 的
   `COPY repos/...` 会带上；再加一个 grep 断言防止回退）
3. 下一个窗口：完整跑完 `offline_inference` 拿输出 + `test_offline_inference` /
   `test_kl` / `test_sampler`、prefix-cache A/B、OpenAI/Anthropic 层

---

## 镜像已重建并核对（16:38 - 16:45，`BUILD_EXIT=0`）

`docker/images`: `deepinfer-sm90:v41-sm90` = `v41-sm90-20261022-1638`（46.2 GB）

```
#16 PR#99 NVLink Engram fetch present     <- Dockerfile 里的断言
#20 SELF-CHECK OK                          <- 镜像原有 7 项自检
```

新镜像内含物核对（在 `--rm` 容器里逐项验证）：

| 项 | 结果 |
|---|---|
| `deepinfer/model_executor/parallel_utils/engram_nvlink.py` | present |
| `userbuffer.py` 的 NVLink 分支 | 2 处 |
| `engram.py` 的 `ENGRAM_NVLINK_FETCH` | 1 处 |
| `libmlx5.so.1` | `1.25.54.0`（rdma-core 54） |
| `mlx5dv_reg_dmabuf_mr` | 1 |
| `tests/distributed/test_engram_nvlink.py` | present |

Dockerfile 新增两段：
- rdma-core 54 构建（`notes/06` 里已论证符号来源）
- PR#99 存在性断言（防止 checkout 回退版本时静默丢补丁）

### 一次构建失败及其真实原因（记录以免误判）

16:30 那次构建 `BUILD_EXIT=1`，卡在 `build.sh` 初始化 `vendor/DeepEP-Ascend` /
`vendor/DeepGEMM-Ascend` 两个子模块时**直连 github.com 超时 135 s**。

我当时以为是子模块没被跳过，加了 `git config submodule.<name>.active=false`。
**16:38 重跑时这两个子模块照样被克隆了**（~1.5 s each，走全局 gh-proxy insteadOf），
`BUILD_EXIT=0`。所以：

- 真实原因是一次性的 github/gh-proxy 网络抖动，**重试即可**；
- `active=false` 拦不住 `git submodule update --init`，那两行只是摆设。
  Dockerfile 里的注释已按实测结论改正，不再声称它能防住这个问题。

教训：CI/构建里的网络抖动要先用"同一命令重跑一次"证伪，再动配置。
