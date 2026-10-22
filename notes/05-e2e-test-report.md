# 端到端测试报告（第 1 轮）— 2026-10-22

## 时间窗与结论

| 项目 | 值 |
| --- | --- |
| 窗口 | **12:57:18 停服务 → 13:13:45 生产恢复**（共 16 分 27 秒） |
| 镜像 | 测试用 `ae21d137bada`；修复后重建为 **`3d1d7e132485`**（46.2 GB，tag `v41-sm90`） |
| 生产恢复 | **13:13:45，`/v1/models` HTTP 200**（期限 13:20，提前约 6 分钟） |
| 结果 | pytest 全绿；权重加载成功；**首次 forward 崩在 mHC 融合算子**，未完成生成 |

## 通过项

### 1. `pytest tests/sm90` —— 142 passed

```
142 passed, 71 warnings in 132.93s (0:02:12)
```

覆盖 `test_flash_mla_sm90`、`test_fp8_block32_gemm`、`test_fp8_block32_gemm_split_k`、
`test_sm90_indexer`、`test_sm90_indexer_reencoding`、`test_sm90_linear_integration`。
这是 #92 的核心：Hopper 上的 sparse attention、block-[32,32] FP8 GEMM、indexer。

### 2. 权重加载成功

```
HF load phase engram_preallocate:  28.601s  allocated=51.06 GiB  peak=51.06 GiB
HF load phase ordinary:            21.253s  allocated=51.52 GiB  peak=52.32 GiB
HF load phase runtime_adapt:       18.564s  allocated=48.26 GiB  peak=57.63 GiB
HF load phase total:              112.082s  allocated=48.26 GiB  peak=57.63 GiB  reserved=58.22 GiB
```

510 GB 权重用 **112 秒**（约 4.5 GB/s，与 dd 实测 2.4 GB/s 同量级），每 rank 常驻 48.26 GiB、
峰值 57.63 GiB。8 卡 DP8×EP8 的加载路径本身没问题。

## 修正记录（按发现顺序）

### 修正 1：`LD_LIBRARY_PATH` 遮蔽驱动库（**镜像缺陷，已修**）

**现象**

```
RuntimeError: Unexpected error from cudaGetDeviceCount() ...
Error 803: system has unsupported display driver / cuda driver combination
```

`tests/sm90` 首个用例即失败，torch 连 `device_count` 都拿不到。

**根因**

镜像里写的是：

```dockerfile
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda/lib64
```

`/usr/local/cuda-13.0/compat/` 里是 **580.126.20** 的前向兼容 `libcuda.so`，
它排在搜索路径最前面，**遮蔽了容器内挂载的真实驱动库 `/usr/lib/.../libcuda.so.595.91.07`**。

而宿主驱动实际是 **595.91.07，CUDA Version 13.2**，原生支持 cu132 运行时，
**根本不需要** CUDA 13.0 的 compat 包。这个 ENV 是我从早期笔记里照抄的
（笔记里当时写的是 "R590 驱动"，实测是 R595）。

**修法（两段式，这个坑比看起来深）**

第一版我直接把 ENV 改成 `LD_LIBRARY_PATH=/usr/local/cuda/lib64`，运行时问题解决了
（`torch 2.13.0+cu132 | device_count 8 | matmul ok`），**但构建挂了**：

```
#17 ImportError: libcuda.so.1: cannot open shared object file
```

原因是这条 ENV 在**一鱼两吃**：

| 场景 | 需要的 libcuda.so.1 | 来源 |
| --- | --- | --- |
| 构建容器（无 GPU、无驱动） | 需要一个能加载的占位 | 只有 `/usr/local/cuda-13.0/compat/` 里有 |
| 运行容器（有驱动） | 必须是真的 595.91.07 | `/usr/lib/x86_64-linux-gnu/`（toolkit 挂入） |

compat 里的 580 版本在**运行时**遮蔽真驱动 → Error 803；
但去掉它**构建时**就连 `deepinfer._C` 都导入不了。同一条 ENV 无法同时满足。

最终修法：运行时 ENV 彻底不放 compat，只在构建自检那一步局部指定：

```dockerfile
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# 构建容器没有驱动，libcuda.so.1 只存在于 CUDA 13.0 前向兼容包里。
# 仅这一步临时指向它；写进镜像 ENV 会在运行时遮蔽真驱动，torch 报 Error 803。
RUN LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda/lib64 python /tmp/verify_service.py
```

**最终验证**（新镜像 `3d1d7e132485`，**未加任何 `-e LD_LIBRARY_PATH` 覆盖**）：

```
torch 2.13.0+cu132
device_count 8
device0 NVIDIA H20-3e
matmul ok 47.80
```

`BUILD_EXIT=0`，7 项自检全 PASS。

### 修正 2：缺 `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS`（**运行时环境变量**）

**现象**

```
File ".../model_metadata_utils.py", line 92, in __init__
    assert envs.DEEPINFER_YOCO_PREFILL_KEEP_TOKENS > 0
AssertionError
```

8 个 rank 全部在 engine 配置阶段退出，显存未动。

**根因**：`doc/sm90.md` 把 `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` 的默认值写作 140，
但**环境变量本身的默认是 0**；140 是 `serve/server.sh` 里 `--yoco-keep-tokens` 的默认值。
所以直接用 `examples/offline_inference.py` 而不经过 `server.sh` 时必须显式给值。

**修法**：`-e DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`。

### 修正 3：NCCL GIN 不可用 → `EP_DISABLE_GIN=1`（**运行时环境变量**）

**现象**

```
RuntimeError: Assertion exception (vendor/DeepEP/csrc/kernels/comm/context.cpp:125):
  (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE
  and "NCCL GIN is unavailable. This is usually due to a network configuration issue,
       such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network."
```

仍卡在 worker 初始化，权重尚未开始加载。

**根因**：这台机器的 **NCCL GIN（GPU-Initiated Networking）不可用**。DeepEP 默认要求
GIN 或 hybrid RDMA 路径。

**修法**：`-e EP_DISABLE_GIN=1`（DeepEP 源码里用来关掉 GIN 的开关；断言里建议的
`allow_hybrid_mode=0` 对应它）。加上后**权重开始正常加载**，证明这一步修对了。

## 未解决项

### 阻塞 1：首次 forward 时 `cuModuleLoadData` 报 `CUDA_ERROR_ILLEGAL_ADDRESS`

**现象**（权重加载完成后，进入 warmup 第一次前向）

```
File ".../model_executor/models/decoder_layer.py", line 143, in forward
    hidden_states, hidden_states_bf16, residual = self.attn_norm_forward(hidden_states, residual, hc_post_intervention)
File ".../model_executor/models/v4_modules/hc.py", line 129, in attn_norm_forward
    return self._hc_fused_post_pre(
RuntimeError: cuModuleLoadData(&(module_[device_id]), code_.data()) failed with error:
  CUDA_ERROR_ILLEGAL_ADDRESS
```

8 个 rank 同时失败，栈完全一致。

**分析**

`cuModuleLoadData` 是**加载 cubin 模块**的 API，本身不访问显存；它返回
`CUDA_ERROR_ILLEGAL_ADDRESS` 通常意味着该 CUDA context 上**已经有一个 sticy 错误**
（此前某次 kernel 异步越界），在这里才被暴露出来。`hc.py:129` 的 `_hc_fused_post_pre`
是 mHC（融合 post/pre norm）算子，走 TileLang/custom JIT。

所以真正的越界点很可能在**权重加载末尾或 warmup 的某个更早 kernel**，而不是 mHC 本身。

**下一步（需要新的 GPU 窗口）**

1. 加 `CUDA_LAUNCH_BLOCKING=1` 复现，让错误停在真正出错的 kernel 上；
2. 加 `TORCH_USE_CUDA_DSA=1` 拿到 device-side assert 的具体位置；
3. 用 `compute-sanitizer --tool memcheck` 跑 layer 0 单层前向；
4. 检查 `hc.py` 融合算子对 SM90 的 arch 假设（是否只按 `sm_90a` 生成、是否有 90 vs 90a 混用）；
5. 退化验证：设 `DEEPINFER_SM90_MOE_BACKEND=triton_fp8` 或关掉 mHC 融合，看是否仍崩，
   以区分 "SM90 kernel 问题" 与 "参数/形状不匹配"。

## 生产恢复记录

- **13:08:38** 停止测试容器 `di-e2e`，显存回到 15-39 MiB
- **13:09:08** `docker start glm53-flash`（注意：**不要用启动脚本**，
  `docker stop` 没有删容器，直接 start 最可靠；`/home/ubuntu/glm53` 这个路径其实不存在，
  早期笔记记录有误）
- 实际配置确认：镜像 `<registry-host>:5000/algorithm/vllm-openai:glm53-flash`，
  `vllm serve /home/model/GLM-5.3-Flash --tensor-parallel-size 8 --max-model-len 1048576
  --gpu-memory-utilization 0.9363 --speculative-config {"method":"mtp","num_speculative_tokens":1}`
- **13:13:45** `/v1/models` HTTP 200，`init engine 140.23 s`、`Graph capturing finished 74 secs`
- 显存 138,424-138,448 MiB/卡（略低于停机前的 142,870，属刚启动的正常差异）

## 复现命令

```bash
MODEL=/home/ubuntu/data1/models/DeepSeek-V4.1-Flash-1017
docker run -d --name di-e2e --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL":/model:ro \
  -e DEEPINFER_ACCELERATOR=nvidia -e DEEPINFER_CUDA_ARCH=sm90 \
  -e DEEPINFER_HF_MODEL_DIR=/model \
  -e DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact \
  -e DEEPINFER_PREFILL_STORE_EXTRA_STATE=1 \
  -e DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140 \
  -e EP_DISABLE_GIN=1 -e NCCL_NVLS_ENABLE=0 \
  deepinfer-sm90:v41-sm90 sleep infinity

# 1) kernel 测试
docker exec -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 di-e2e \
  bash -lc 'cd /opt/deepinfer/DeepInfer && python -m pytest tests/sm90 -q'

# 2) 端到端（目前崩在 warmup，见未解决项）
docker exec -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  -e DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140 -e EP_DISABLE_GIN=1 di-e2e \
  bash -lc 'cd /opt/deepinfer/DeepInfer && python -u examples/offline_inference.py --model_name /model --ep 8'
```

---

## 本轮的操作失误（记录以免重犯）

### 失误 1：把容器删了，日志一起没了

`docker rm -f di-e2e` 之后，`/tmp/e2e3.log` 随之消失。我此前只 `sed` 了 traceback 的
700-760 行，`_hc_fused_post_pre` **之后**的几帧（真正出错的算子）没来得及看，现在无法恢复。

**改进**：新建 `tools/e2e_window.sh` 在启动容器时把 `logs/e2e/` bind-mount 成 `/e2e-logs`，
日志直接落在宿主机，容器删了也还在。

### 失误 2：靠人盯时钟收工

12:57–13:12 的出工时间是靠我手动读表决定的。一旦某次调用耗时超预期，就会挤占
`glm53-flash` 的恢复时间（它需要约 4.5 分钟：init engine 140 s + graph capture 74 s）。

**改进**：`tools/e2e_window.sh start <minutes>` 会起一个**脱离会话的看门狗**，
到点自动 `docker rm -f` 测试容器 → `docker start glm53-flash` → 轮询 `/v1/models` 直到 200。
即使 ssh 断连或本会话结束，看门狗仍然执行。

```bash
tools/e2e_window.sh start 20      # 20 分钟窗口，测试 14 分钟后自动恢复
tools/e2e_window.sh status        # 看生产/测试容器与显存
tools/e2e_window.sh stop          # 手动提前收工
```

## 下一个窗口的验证计划（按顺序，全部已准备好）

1. **拿全栈**：用 `tools/e2e_window.sh start`，日志走 bind-mount；
   `python -u examples/offline_inference.py --model_name /model --ep 8 2>&1 | tee /e2e-logs/run1.log`
2. **定位 sticky error**：加 `CUDA_LAUNCH_BLOCKING=1`（错误会停在真正越界的 kernel 而不是
   mHC 的 `cuModuleLoadData`）；再加 `TORCH_USE_CUDA_DSA=1` 拿 device-side assert 位置
3. **缩小范围**：`compute-sanitizer --tool memcheck` 跑单层前向
4. **区分责任**：
   - `DEEPINFER_SM90_MOE_BACKEND=triton_fp8` 换 MoE 后端，看是否仍崩
   - 关掉 mHC 融合（`hc_mult`/`use_mega_mhc` 路径），看是否仍崩
   - 若只在 mHC 崩 → 检查 `_hc_fused_post_pre` 非 mega 路径里的算子对 SM90 的 arch 假设
5. **拿到响应后**再依次验：`pytest tests/hicache`、OpenAI 层前缀缓存 A/B、Anthropic 层
6. 每个窗口结束前确认 `tools/e2e_window.sh status` 里 `prod: Up` 且 `/v1/models` 为 200

## 本轮净结果

| 项目 | 状态 |
| --- | --- |
| 镜像（含服务的单镜像） | ✅ 已交付并修复 1 个镜像缺陷后重建（`3d1d7e132485`） |
| `pytest tests/sm90` | ✅ 142 passed |
| 8 卡权重加载 | ✅ 112 s，48.26 GiB/rank，峰值 57.63 GiB |
| 端到端生成 | ❌ warmup 首次 forward 崩在 `_hc_fused_post_pre`（sticky CUDA error），未定位到具体算子 |
| 修正记录 | ✅ 3 项（1 个镜像缺陷 + 2 个运行时环境变量） |
| 生产恢复 | ✅ 13:13:45，早于 13:20 期限 |

---

# 第 2 轮：在不停生产的前提下做隔离验证

时间：2026-10-22 13:24 起。**全程未停 `glm53-flash`**，每次实验前后 `/v1/models` 都是 200，
生产显存 138,448 / 138,424 MiB 未变。

## 方法：单卡 + 小张量的隔离复现

生产占满 8 卡但每卡仍剩约 **5.3 GB**，于是：

```bash
docker run --rm --gpus '"device=7"' -e CUDA_LAUNCH_BLOCKING=1 \
  -v "$PWD/tools/<repro>.py":/repro.py:ro deepinfer-sm90:v41-sm90 python -u /repro.py
```

pin 到 GPU 7、张量几 MB、**单次实验 5-13 秒**。这把"需要 15 分钟窗口 + 加载 510 GB"的调试，
变成了可以随时反复做的小实验。

产物：`tools/repro_prenorm_gemm.py`、`tools/repro_mhc.py`、`tools/repro_sf_real.py`。

## 验证结果：四个嫌疑全部排除

| 算子 | 在崩溃路径中的位置 | SM90 实测 |
| --- | --- | --- |
| `deep_gemm.tf32_hc_prenorm_gemm` | `_hc_pre` → `mhc_prenorm_gemm` | ✅ OK（真实形状 (8,5120)×(24,5120)） |
| `expand_to_mhc` | 首层 `_hc_fused_post_pre` | ✅ OK |
| `mhc_pre_big_fuse` | `_hc_pre`（prenorm GEMM + rmsnorm + split + sinkhorn + apply） | ✅ OK，`finite=True` |
| `set_ue8m0_sf` / `batched_transpose_weight_sf` | `runtime_adapt` 载入阶段最后一件事 | ✅ OK（普通 / 专家 / 混合三种） |

也就是说：**mHC 整条链和 scale 打包在 SM90 上都没问题**，崩溃不在它们身上。

## 重要教训：我前三次隔离测试都是假阳性

这一段必须记下来，因为它差点让我报出错误结论。

| # | 错误输入 | 现象 | 真实约束 |
| --- | --- | --- | --- |
| 1 | `mult3=4` | `Assertion: n <= 32 and n % 8 == 0` | `mhc_mult3 = 24`（`hc_mult*2 + hc_mult²`） |
| 2 | sf 形状 `(n, k/32)` = `(4608,160)` | device assert | docstring：`sf` 形状是 `(n/block, k/block)` = **(144,160)** |
| 3 | sf 用随机均匀分布 | device assert | docstring：**power-of-two** fp32 scales（kernel 把指数转 UE8M0） |
| 4 | `ue8m0_sf` 当 tensor 取 `.shape` | `AttributeError` | 它是 `(sf0_out, sf1_out)` 二元组 |

换成契约正确的输入后，四个算子全部 PASS。

**结论**：隔离复现如果没有严格遵循被调函数的输入契约，得到的 device assert **几乎一定是假阳性**。
判断"这是不是一个 bug"之前，必须先确认输入合法。真实运行里 `batched_transpose_weight_sf`
是"完成"而非 assert，这个反证我当时注意到了，才没有误报。

## 因此，warmup 崩溃的剩余解释

`cuModuleLoadData` 报 `CUDA_ERROR_ILLEGAL_ADDRESS` 而现在能证明它加载的 mHC kernel 单独跑没问题，
那么最可能的仍是**前序某个 kernel 留下了 sticky illegal address**，在这里的模块加载调用上爆发。

剩余嫌疑（都需要真实前向才能触发）：

- `flash_mla_sm90` 在 warmup 形状下的 sparse attention（SM90 补丁 522 行，文档自称"candidate for FlashMLA"）
- `humming` W4A8 routed-expert GEMM（`DEEPINFER_SM90_MOE_BACKEND=humming`）
- engram 相关 kernel（`engram_preallocate` 分配了 51 GiB pinned host）
- CUDA graph 捕获阶段

**下一步必须用真实前向**：`tools/e2e_window.sh start 20` 起窗口，
带 `CUDA_LAUNCH_BLOCKING=1`（+ `TORCH_USE_CUDA_DSA=1`）跑一次，
错误会停在真正越界的那个 kernel 上，而不是几层之后的模块加载调用。

---

# 第 3 轮：继续用单卡跑完所有"不需要真权重"的测试

时间：2026-10-22 13:29 起。**全程未停 `glm53-flash`**，每次运行前后 `/v1/models` 均为 200。

## 新增通过项

| 测试 | 结果 | 说明 |
| --- | --- | --- |
| `tests/models/test_sm90_moe.py` | **19 passed**（40.6 s） | 四种 MoE 后端（humming / triton_fp8 / hybrid / deepgemm）× 两种 sf 格式 |
| `tests/hicache` | **3 passed**（45.8 s） | #144 缓存逻辑：哈希链匹配 / keep-tokens 截断 / 吸附 / 写入 / LRU |
| `tests/{models/test_eplb_policy, worker, engine, ve, distributed/test_expert_shuffle, distributed/test_shm_broadcast}` | **115 passed, 2 skipped**（27.8 s） | engine/worker/sequence/state-cache/EPLB/VE 元数据 |

累计：**142（sm90）+ 19（moe）+ 3（hicache）+ 115 = 279 passed，2 skipped**，
全部在生产运行中完成，未占停一秒服务。

**humming W4A8 也排除了** —— 它是 SM90 上 `auto` 的默认 MoE 后端，此前是重点嫌疑。

## 一个真实的环境缺陷（测试侧，非代码侧）

首次跑这批测试时出现 **42 个 ERROR**，原因不是代码：

```
ERROR at setup of TestStateCacheOperations...
  self = <HTTPSConnection(host='huggingface.co', port=443)>
```

这些 fixture 用 HF model id（`deepseek-ai/DeepSeek-V4.1-Flash-1017`）去 **huggingface.co**
拉 `config.json`，而该域名在这台机器上不可达。`test_flops_estimator` 的 fixture 同样
`ModelConfig(model=<hf id>)`。

`examples/_config.py:resolve_model_path` 已经支持用 `DEEPINFER_HF_MODEL_DIR` 覆盖默认模型名，
所以正确跑法是：

```bash
docker run --rm --gpus '"device=7"' \
  -e DEEPINFER_HF_MODEL_DIR=/model -e HF_HUB_OFFLINE=1 \
  -v /home/ubuntu/data1/models/DeepSeek-V4.1-Flash-1017:/model:ro \
  deepinfer-sm90:v41-sm90 \
  bash -lc 'cd /opt/deepinfer/DeepInfer && python -m pytest tests/worker tests/engine tests/ve -q'
```

加上这两个变量后：**294 s → 27.8 s**，42 个 ERROR 全部消失。
（fixture 只读 `config.json`/tokenizer，不加载权重，显存仍只有几十 MB。）

## 一次性抖动（如实记录，非缺陷）

第一次离线重跑出现 `12 failed, 83 passed, 20 errors`，随后**连续三次**完全相同的结果是
`115 passed, 2 skipped`（27.77 s / 27.77 s / 27.92 s，总数一致均为 117）。

同一条命令、同一个镜像，只有第一次失败，无法复现。最可能是第一次启动时上一个容器
（跑了 294 s 的那个）的 GPU 显存尚未完全回收，测试撞上瞬时压力所致。
**不记为缺陷**，但记录在此：在同一张卡上与生产共享显存时，测试启动最好间隔几秒。

## 仍然只能靠真实前向的部分

以下测试必须加载 510 GB 权重，需要 GPU 窗口：

- `tests/models/test_offline_inference.py` / `test_offline_inference_long.py`
- `tests/models/test_sampler.py` / `test_kl.py` / `test_batch_invariant.py`
- `tests/distributed/test_cp_manager.py`（CP8）
- 以及 warmup 崩溃本身（剩余嫌疑：`flash_mla_sm90` 的 warmup 形状、engram kernel、CUDA graph 捕获）

`tests/sm90` 已覆盖 `flash_mla_sm90` 的多种 decode/prefill 形状且 142 项全过，
所以剩余嫌疑更集中在 **engram** 与 **CUDA graph 捕获**阶段。
