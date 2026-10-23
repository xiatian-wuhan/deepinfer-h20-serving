# 笔记索引

26 篇笔记是这条链路的完整推演过程：从"跑不起来"到"三协议 + 缓存命中"。按阶段分组，
每条一句话结论 + 状态：

- **现行** —— 当前结论的出处，改动前该读。
- **历史** —— 保留现场和推理过程，其中的数字可能已被后续测量取代（正文会标出取代关系）。

只想快速上手：性能看 [`16`](16-cp8-layout.md)/[`17`](17-engram-fetch-fix.md)/[`18`](18-cp8-for-1m.md)，
接口看 [`25`](25-recipe-protocol-layer.md)，两个都在 [`README`](../README.md) 里有汇总表。

## A. 环境与镜像（01–04）

| # | 主题 | 结论 | 状态 |
|---|---|---|---|
| [01](01-deps.md) | 依赖分析与镜像选型 | 走 #92 的 `build.sh`，基座 `lmsysorg/sglang:v0.5.22` | 历史 |
| [02](02-prefix-cache.md) | 上游前缀缓存方案盘点 | 有三级实现都没合主线，PR #144 可与 #92 叠加 | 现行 |
| [03](03-build-log.md) | 两步构建的中间镜像 | 中间产物已删，六个坑对单镜像依然适用 | 历史 |
| [04](04-final-image.md) | 交付镜像 `deepinfer-sm90:v41-sm90` | 单镜像 46.2 GB，验收清单在此 | 现行 |

## B. Engram / GIN 排障（05–09、21、23）

| # | 主题 | 结论 | 状态 |
|---|---|---|---|
| [05](05-e2e-test-report.md) | 第 1 轮端到端测试 | 功能面通过，warmup 阶段崩在 Engram | 历史 |
| [06](06-engram-gin-rootcause.md) | 崩溃根因 | Engram 取表依赖 NCCL GIN，GIN 不可用即崩 | 现行 |
| [07](07-iommu-pt-changeset.md) | 宿主改 `iommu=pt` 变更单 | 需要重启的变更，含回滚 | 历史（已执行） |
| [08](08-nvidia-escalation.md) | NVIDIA 工单材料 | H20-3e + R595 下 GPUDirect RDMA 不可用的取证 | 历史 |
| [09](09-pr99-nvlink-breakthrough.md) | PR #99：绕开 GIN 的取表路径 | 不停生产就解决了崩溃 | 现行 |
| [21](21-gin-fixability.md) | GIN 到底能不能修 | 能，但必须换内核模块形态（专有 → 开源） | 现行 |
| [23](23-driver-swap-gin.md) | 换 `nvidia-driver-595-open` | GIN 点亮，回滚材料在 `/root/driver-swap/` | 现行 |

## C. 服务层与性能口径（10–13）

| # | 主题 | 结论 | 状态 |
|---|---|---|---|
| [10](10-serve-layer-report.md) | 服务层首份报告 | 当时 14 项断言；现在是 `tools/serve_test.py` 的 26 项 | 历史 |
| [11](11-openai-protocol-audit.md) | 协议完备性审计（对照 vLLM） | 逐条列出 vLLM 有而我们没有的字段 | 现行 |
| [12](12-perf-comparison-correction.md) | 性能对比口径更正 | 参考数字的测法、并发口径、预填配置都不同，不能直接比 | 现行 |
| [13](13-perf-dp8-official.md) | DP8 用官方 harness 实测 | 数字已被 `README §5` 的同一 harness 重测取代 | 历史 |

## D. 布局与 Engram 优化（14–18）

| # | 主题 | 结论 | 状态 |
|---|---|---|---|
| [14](14-sglang-engram-design.md) | SGLang 怎么在 8×H20 上跑 | 那套优化不在可用版本里；Engram 是差距所在 | 历史 |
| [15](15-profiler-engram-breakdown.md) | Profiler 归因 | 单步 59 ms 里 71% 是 Engram 的 NVLink 取表 | 现行 |
| [16](16-cp8-layout.md) | CP8 复现 | 布局选错，不是代码慢；CP8 把单会话解码拉高一个量级 | 现行（结论）/ 数字见 README |
| [17](17-engram-fetch-fix.md) | Engram fetch 去冗余 | padding 从 32768 降到真实 token 数，DP8 解码 86→152 | 现行 |
| [18](18-cp8-for-1m.md) | 1M 上下文用哪种布局 | 1M 只有 CP8 可用（快 4.7 倍） | 现行（结论）/ 数字见 README |

## E. 网关、工具调用与协议层（19、20、22、24、25）

| # | 主题 | 结论 | 状态 |
|---|---|---|---|
| [19](19-cache-usage-through-gateway.md) | 经网关时"缓存↓"不显示 | 网关只认上游 usage，服务端必须无条件上报 | 现行 |
| [20](20-tool-calls-dsml.md) | DSML 工具调用解析 | 手写解析的三个坑（过严的探测标记） | 历史（解析已交给 recipe） |
| [22](22-message-order-verification.md) | 消息顺序第 1 轮 | 证明我们这边不重排 | 历史 |
| [24](24-message-order-vs-official-encoding.md) | 和官方两份实现对表 | 逐 token 对表的方法与结果 | 现行（方法） |
| [25](25-recipe-protocol-layer.md) | 协议层换成 deepseek-recipe | 渲染/解析/流式都归官方 Rust 实现，我们只做引擎对接 | 现行 |

## F. 选型评估（26）

| # | 主题 | 结论 | 状态 |
|---|---|---|---|
| [26](26-qwen36-on-deepinfer-feasibility.md) | Qwen3.6-35B-A3B 能否用 DeepInfer 推理 | 能改，但等于新写一个模型栈（80–190 人日）；不建议，vLLM/SGLang 已原生支持 | 结论（2026-10-23） |

## 数字在哪

笔记里的表都是当时那一跑的快照。当前对外口径只有一个地方：`README §五`（性能）
和 `§6.7`（缓存命中），数字由 `data/` 下冻结的运行结果生成，跑法写在 `data/README.md`。
