# 工具索引

分三类：**日常复用**（做完改动就该跑）、**宿主运维**（要 root 或重启窗口）、
**留档**（一次性排障脚本，结论已写进 notes，脚本只为可复现）。

所有"服务层"脚本默认打 `http://127.0.0.1:8888`，可用 `DEEPINFER_SERVE_BASE`
覆盖；模型名默认 `DeepSeek-V4.1-Flash-1017`，可用 `DEEPINFER_SERVED_MODEL_NAME` 覆盖。

## 一、协议与服务层（改动后必跑）

| 脚本 | 作用 | 跑法 |
|---|---|---|
| `serve_test.py` | 26 项协议断言：非流式/流式、usage、cached_tokens、三协议、图片、thinking、count_tokens | `python3 tools/serve_test.py` |
| `serve_bench.py` | 服务层 TTFT/TPOT/单流与聚合吞吐；`--no-cache` 做冷预填 A/B | `python3 tools/serve_bench.py --concurrency 8 --count 32` |
| `cache_hit_matrix.py` | 三个协议各自缓存字段是否都上报、是否同一个数、是否跨协议共享 | 容器内 `python /tmp/cache_hit_matrix.py`（见下） |
| `recipe_parity.py` | deepseek-recipe 与本机 checkpoint 自带编码器/解析器逐项对表（A1/A2 提示词与 token id、A3 控制符注入、A4 图片占位、C 输出解析） | 容器内 `python /tmp/recipe_parity.py` |
| `check_encoding_parity.py` | 消息顺序：同一批请求分别走服务层与 `encoding.py`，逐 token 比对 | 容器内 |
| `live_order_probe.py` | 活体探针：模型看到的和吐出的顺序是否一致（不依赖离线对表） | `python3 tools/live_order_probe.py` |
| `analyze_trace.py` | 扫真实 Claude Code trace（`serve.log`）里的顺序不变量 | `python3 tools/analyze_trace.py serve.log` |

服务层脚本要在容器里跑，是因为服务只监听容器内回环：

```bash
docker cp tools/cache_hit_matrix.py di-e2e:/tmp/ && docker exec di-e2e python /tmp/cache_hit_matrix.py
docker cp tools/recipe_parity.py   di-e2e:/tmp/ && docker exec di-e2e python /tmp/recipe_parity.py
```

## 二、宿主与驱动运维

| 脚本 | 作用 | 备注 |
|---|---|---|
| `verify_gin.sh` | 20 秒 GIN 可用性探针，不碰生产、不加载权重 | 换驱动/换内核后第一件事 |
| `post_reboot_check.sh` | 重启后四步验收，一条命令给 PASS/FAIL | 搭配上面的探针 |
| `e2e_window.sh` | 测试窗口 + 硬看门狗（超时自动恢复生产） | **当前看门狗已按用户要求关闭**，恢复时机由用户决定，不要自动启用 |
| `probe_dmabuf.py` | 验 `cuMemCreate → cuMemMap → DMA_BUF_FD` 能否真的拿到 fd | 判定 GPUDirect RDMA 是否可用 |
| `patch_gin.py` | 改 `deep_ep` 里 `NCCL_GIN_TYPE` 默认值，用于 GIN 路线验证 | 一次性 |
| `_apply_iommu_pt.sh` | 宿主 GRUB 加 `iommu=pt`，含备份与校验 | 已执行（notes/07） |
| `_pin_kernel_57.sh` / `_restore_default_pin.sh` | 把 GRUB 默认项钉到运行中的内核 / 还原运维原有写法 | 已执行 |
| `build_rdma_and_probe.sh` / `build_rdma_core_test.sh` | 源码编译 rdma-core 覆盖系统库并跑 GIN 探针 | 历史路线：最终改用开源驱动模块解决 |

## 三、性能剖析

| 脚本 | 作用 |
|---|---|
| `prof_bench.py` | 在 `examples/latency_benchmark.py` 的 single 套件上挂 `torch.profiler`，抓 rank0 的若干解码步，直接打印 kernel 占比（notes/15 的出处） |
| `prof_prefill.py` | 同上，抓预填阶段 |

## 四、留档（一次性最小复现，已结案）

| 脚本 | 复现的问题 |
|---|---|
| `repro_prenorm_gemm.py` | warmup 崩溃：`deep_gemm.tf32_hc_prenorm_gemm` 在 SM90 上 |
| `repro_sf_real.py` | 同一崩溃链上的 `set_ue8m0_sf()` |
| `repro_mhc.py` | 用 checkpoint 真实形状逐算子走 mHC 链 |
| `old_marker_demo.py` | 旧 DSML 探测标记在合法布局上的误判（解析现已交给 deepseek-recipe） |
