# 补丁说明

## 唯一需要应用的

`deepinfer-h20-full.patch`：相对上游 `DarkraiHL/DeepInfer` 的 `e205d42`（分支 `sm90-hopper`）的完整改动，
**19 文件，+2601 / −1**。已在干净的上游基线上跑过 `git apply --check` 并真实应用过一次，
应用后的工作树就是本仓库记录的那个状态。

```bash
git clone https://github.com/DarkraiHL/DeepInfer.git
cd DeepInfer
git checkout e205d42e4b28185e50ce30f6b7204cd01e59974c
git apply /path/to/deepinfer-h20-full.patch
```

## 内容构成

补丁里混合了上游尚未合并的 PR 和我们自己的改动，按主题：

| 主题 | 说明 |
|---|---|
| SM90 支持（#92） | H20-3e 的算子与调度路径 |
| 前缀缓存（#144） | `deepinfer/hicache/` host KV 页池；配套的 scheduler 绑定来自 #195 |
| 服务层（#147） | `serve/deepseek_server.py`：三协议、流式、图片、工具调用、缓存命中上报 |
| NVLink Engram fetch（#99） | `engram_nvlink.py`：GIN 不可用时的取数路径 |
| **本仓库的改动** | Engram fetch 去冗余 + 缓冲按容量复用（README §七）、`--cp` 透传、图片输入、模型名可配、协议层交给 deepseek-recipe（`notes/25`） |

## history/

中途生成的分解视图（按上游 PR 分开），**不要单独应用**。原因：

- 它们是不同时刻的快照，和最终的 `deepinfer-h20-full.patch` 对不上——`pr144` 里还带着后来回退的
  `vendor/DeepEP` 子模块指针和 `scripts/build_vendor.sh`，而 scheduler 绑定（`pr195` 的内容）
  没有落在 `pr144` 里，只在 `full` 里；
- 快照里服务入口还叫 `serve/openai_server.py`，2026-10-23 已改名 `serve/deepseek_server.py`。

留档只为看清"每个 PR 各贡献了哪一块"。
