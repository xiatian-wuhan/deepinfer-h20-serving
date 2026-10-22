# DeepInfer 的前缀缓存方案盘点

## 结论

**有，共三个层次，都没合进主线。其中 PR #144 与 #92 实测可直接叠加。**

## 方案 1：PR #144 — host 前缀缓存（HiCache 风格）★ 最接近可用

- 规模：10 文件 / 720 行（含 190 行测试）
- 核心：`deepinfer/hicache/host_cache.py` (437 行) + `kernels.py` + `page_copy.cu/.cpp`
- 布局：两块 pinned host 池，按页组织
  - KV 页：一页 = 一个 paged-KV block（block_size tokens，全层），与 GPU block 等大
  - State slots：序列的 SWA "stored region"，在 128 对齐位置取
- 索引：**hash 链**（page i 的 key 覆盖 tokens [0, (i+1)*block_size)），查找走链直到首个 miss
- 淘汰：**LRU**，两池独立
- 传输：小 CUDA kernel 走 UVA 指针，在引擎当前 stream 上，CUDA event 轮询
- 策略：每次 prefill 结束 **write-through**；新请求取最长页链匹配，但保留 `keep_tokens`
  （decoder bounded replay 区），对齐到最近有 state 的位置，带 `num_cached_tokens` 进调度器
- 前置：`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`
        `DEEPINFER_PREFILL_STORE_EXTRA_STATE=1`
- 未覆盖：decode 产生的 token 不存；**缓存是每引擎进程的**（同一会话多轮需路由到同一 rank）

### 与 #92 的兼容性（本工作区实测）

```
git apply --check --exclude=scripts/build_vendor.sh patches/pr144-prefix-cache.patch
→ ✅ 9/10 文件干净应用
```

唯一"冲突"的 `scripts/build_vendor.sh` 里，那段 `GIT_REWRITE` 改动 **#92 的初始提交
d585e65 就已存在**，属重复应用，排除即可。

**结论：#144 可以直接叠在 #92 上，无需改代码。**

## 方案 2：deepinfer-server RFC 0001 — 服务层精确前缀缓存

- 状态：**Proposed**（2026-10-21），未实现
- 两阶段：
  1. 服务端保留已完成的 DeepInfer sequence，通过 `Scheduler.fork_seq` 复用（RFC 认为可放本仓库）
  2. 引擎级内容寻址缓存（RFC 认为"属上游 DeepInfer，应作为独立项目"）
- 现状：README 明说"每轮重新 prefill 全上下文"

## 方案 3：deepinfer-serving — 共享 KV store + hash 链前缀匹配

- `src/deepinfer_serving/kvstore.py` 有前缀匹配
- 设计哲学不同：ADR `0001-shared-kv-cache-no-prefix-affinity`
  —— 用**共享 KV cache** 让所有节点等代价访问，路由**不需要前缀亲和**
- 未接真引擎（milestone 5 待做）

## 底层原语：fork_seq

`deepinfer/_C/engine.pyi:316`

```
fork_seq(seq, blocks, target_branch_id, sampling_params,
         so_matcher, sp_token, replace_last_token_id, seq_id, arrival_time) -> Sequence
```

作用：pin 源 sequence 的 KV blocks 和 state、克隆 state、复用已算前缀。
**这是三个方案共同的底层基础。**

## 与 DeepSeek 生产架构的差别

生产 Q&A 明确说：Block KVCache 用**独立 3FS 存储集群**，"使得所有 prefill 机器能命中
任何前缀，避免前缀亲和和负载均衡路由策略冲突"。

- #144 = 单机 host 内存层（HiCache 风格），是**单机简化版**
- carlulsoe = 共享本地 NVMe + 跨引擎 pull
- 生产 = 3FS 集群，全节点等代价访问

对**单机 8 卡 H20** 场景，#144 这个层次足够（不涉及跨节点路由），
只需把同一会话的多轮请求哈希到同一 rank。

## 本工作区已就绪

```
repos/deepinfer-main/  已 fetch 分支 pr-144 / pr-147 / pr-195 / pr-197
patches/pr144-prefix-cache.patch      805 行
patches/pr147-openai-server.patch     OpenAI 兼容 server
patches/pr195-scheduler-api.patch     Scheduler.add_prompted_sequence 等
patches/pr197-pd-server.patch         P/D 跨机分离
```
