# 交付镜像：deepinfer-sm90:v41-sm90

## 一个镜像，不是两个

**交付物只有一个镜像。** 早期我做过两步构建（`Dockerfile.sglbase` 引擎层 → `Dockerfile.service`
用 `FROM` 叠服务层），那是我按任务先后顺序干活的产物，不是设计需求，已废弃
（见 `docker/legacy/README.md`）。两步搭建有两个实际问题：最终镜像依赖一个可变的中间 tag；
而且 vendor wheel 和 `deepinfer._C` 分别来自 p144 树和 p195+p147 树，不是同一 commit 的产物。

现在的 `docker/Dockerfile` 一次到底，`./build.sh` 在同一次运行里产出 vendor wheel 和 `_C`。

## 镜像

| 项目 | 值 |
| --- | --- |
| Tag | **`deepinfer-sm90:v41-sm90`**（另有日期快照 `v41-sm90-20261022-1322`，同一 ID） |
| ID | `3d1d7e132485`（构建于 13:22；端到端测试后修掉 `LD_LIBRARY_PATH` 缺陷并重建） |
| 大小 | 46,179,654,035 B（**46.2 GB**） |
| 构建时间 | 2026-10-22T12:38:50+08:00 |
| 基础镜像 | `lmsysorg/sglang:v0.5.22`（CUDA 13.0.88、Python 3.12.15、gcc 13.3） |
| 源码 | `repos/deepinfer-sm90-92-p195-p147` @ **`885eea2`** |
| GPU 使用 | **0**（纯 CPU 编译，生产 `glm53-flash` 全程未受影响） |

## 源码栈

| 来源 | 内容 |
| --- | --- |
| #92 (`e205d42`) | DarkraiHL 的 SM90/Hopper 移植（Gluon block-[32,32] FP8 GEMM、Triton indexer、FlashMLA-SM90、humming W4A8 MoE） |
| #144 (`6985279`) | host prefix cache：`deepinfer/hicache/`（437 行 `host_cache.py`）+ `Sequence.block_table` |
| #195 | `Scheduler::free_finished_seqs` / `add_prompted_sequence`（PD 分离接口） |
| #147 | `serve/`：OpenAI 兼容层 |
| yiliu30/deepinfer-server | Anthropic Messages 服务 |

## 构建期实测（一次 `./build.sh`）

```
 66.8s  🔧 DeepSelect (CUDA)
151.4s  🔧 FlashMLA-SM90 (CUDA)
259.8s  🔧 DeepEP (CUDA)
315.1s  🔧 DeepGEMM (CUDA)
388.9s  📦 Installing wheels
390.3s  ✅ vendor operators installed
422.4s  ✅ DeepInfer is successfully built
```

7 项自检全部 PASS（自检不用 `|| true`，失败就 fail 构建）：

```
PASS  PR#195 scheduler bindings  (Scheduler.free_finished_seqs + add_prompted_sequence)
PASS  PR#144 prefix cache        (Sequence.block_table; hicache import deferred (no GPU))
PASS  PR#144 hicache sources     (5 files present)
PASS  PR#147 serve layer         (serve/deepseek_server.py imports)
PASS  deepseek-recipe            (Anthropic)
PASS  deepinfer-server engine pool
PASS  serve/ app factory         (build_app)
SELF-CHECK OK
```

## 运行时实测（`--gpus all` + `CUDA_VISIBLE_DEVICES=` 置空）

技巧依据：`deep_cluster/hardware/nvidia.py:compute_capability()` 用 ctypes 直接调 NVML，
docstring 明确 *"no CUDA context is created"*。因此 `--gpus all` 足以让 `is_sm90()` 读到真实能力，
而 `CUDA_VISIBLE_DEVICES=` 置空让 torch 不建 context，**对生产显存零占用**。

```
torch                 2.13.0+cu132
accelerator_kind      nvidia
compute_capability    (9, 0)
is_sm90()             True
torch.cuda.is_avail   False

Sequence.block_table    True      #144
free_finished_seqs      True      #195
add_prompted_sequence   True      #195
HostPrefixCache         导入成功

deep_select 1.0.0   flash_mla_sm90 1.0.0   deep_ep 2.5.0   deep_gemm 2.8.2
humming 0.1.12      flash_attn 4.0.0b34    deepseek_recipe 0.2.0

Anthropic app  DeepInfer Claude Code Gateway
  routes: /v1/messages, /v1/messages/count_tokens, /v1/models, /health
  EP/DP = 4/4, gpus = [0,1,2,3]

显存 142894 / 142870 MiB 未变        生产 glm53-flash HTTP 200
```

这里最要紧的一条是 `HostPrefixCache` 导入成功。构建期它曾报
`ModuleNotFoundError: No module named 'flash_mla'`，原因不是缺陷：
`attention_backend.py` 按 `PLATFORM.is_sm90` 分派，SM100 分支才用 `flash_mla`，
而 `is_sm90` 读 NVML，构建容器无 GPU 时为 False。运行时在真实 H20 上确认走了 SM90 分支，
**前缀缓存可用，不会被静默禁用**。

## 启动

```bash
IMAGE=deepinfer-sm90:v41-sm90
MODEL=/home/ubuntu/data1/models/DeepSeek-V4.1-Flash-1017

# ---- OpenAI 兼容层（#147）----
docker run --rm --gpus all -it \
  -v "$MODEL":/model:ro -p 8888:8888 \
  -e DEEPINFER_HF_MODEL_DIR=/model \
  -e DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact \
  -e DEEPINFER_PREFILL_STORE_EXTRA_STATE=1 \
  $IMAGE serve/server.sh start --dp 8 --port 8888 --routing affinity

#   缓存命中率
#   curl -s localhost:8888/deepinfer/metrics | grep hicache
#   python serve/check_prefix_cache.py --port 8888 --turns 3

# ---- Anthropic Messages 层（deepinfer-server）----
docker run --rm --gpus all -it \
  -v "$MODEL":/model:ro -p 8000:8000 \
  -e GATEWAY_MODEL=/model -e GATEWAY_GPUS=0,1,2,3,4,5,6,7 \
  $IMAGE /opt/deepinfer/deepinfer-server/start.sh
```

前缀缓存要真正命中，三条缺一不可（`host_cache.py` docstring 与 `serve/README.md` 写明）：

1. `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`
2. `DEEPINFER_PREFILL_STORE_EXTRA_STATE=1`
3. `--routing affinity` —— 缓存是 per engine process，同一会话必须固定到同一 rank

## 前缀缓存实现（`deepinfer/hicache/host_cache.py`，437 行）

- **布局**：两个 pinned host 池按页组织。KV 页（一页 = 一个 paged-KV block，含所有层）；
  state 槽（SWA 的 stored region，在 128 对齐处用
  `GPUCacheInterface.select_state_slice_to_store` 取出）。页以 token 哈希链索引，第 i 页 key 覆盖
  `[0, (i+1)*block_size)`，**查找即从第一页逐页走、遇首个 miss 停**；两池各自 LRU。
- **传输**：只用 kernel 方式——小 CUDA kernel 经 UVA 指针在 GPU 池与 pinned host 池间拷贝，
  跑在 engine 当前 stream 上，与模型 kernel 有序，不阻塞 CPU；CUDA event 在 engine loop 轮询。
- **命中**：`try_resume` → `_try_resume`，返回 `0`(miss) / `-1`(命中但 GPU 块暂不可分配，稍后重试) /
  `>0`(命中的 token 数)。先按 `limit = ((n - keep_tokens) // bs) * bs` 截断，再向下吸附到最近一个
  有 `ready` state 的 128 对齐位置；然后 `try_allocate_seq` 占块、refcount + `move_to_end`、
  kernel 拷页并装载 state，最后写 `seq.num_cached_tokens = b` 与 `seq.cached_hit_tokens = b`。
  `min_hit_tokens = 2 * block_size = 256`。
- **写入**：`on_prefill_step` → `_store`，每次 prefill 结束 write-through 存所有满块 + 最后
  128 对齐位置的 state；已存在的页只 `move_to_end`。`DEEPINFER_HICACHE_STATE_LOOKBACK` 默认 4。
- **统计**：`lookups hits hit_tokens lookup_tokens deferred stores store_skipped pages_stored
  pages_loaded states_stored states_loaded evicted_pages evicted_states errors` +
  `hit_rate_tokens` + `pages_used/pages_total` + `states_used/states_total` + `inflight`。
- **容错**：`try_resume` 外层 `except Exception` → 记日志、`_fail()`、返回 0，
  注释写明 *"never take the engine down for a cache problem"*。
- **不覆盖**：decode 产生的 token 不存；缓存 per engine process。

## 尚未验证（需要停掉 glm53-flash）

8 卡各仅剩约 900 MiB，加载 510 GB 权重必须占满 8 卡。以下都没做：

- OpenAI 层真实生成与前缀缓存命中率
- Anthropic 层接 Claude Code 真实对话
- `pytest tests/sm90 tests/hicache`
- `examples/latency_benchmark.py` 复现文档数值（TTFT 2.114 s@128K、c64 4259 tok/s）
