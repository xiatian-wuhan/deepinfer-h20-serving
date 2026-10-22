# 补丁说明

`deepinfer-h20-full.patch` 是**唯一需要应用的补丁**：相对上游
`DarkraiHL/DeepInfer` 的 `e205d42`（分支 `sm90-hopper`）的完整改动，
19 文件，+2549 / −1。

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
| 前缀缓存（#144） | `deepinfer/hicache/`，host KV 页池 |
| scheduler API（#195） | `csrc/engine/` 与 `_C/engine.pyi` |
| OpenAI 服务层（#147） | `serve/`：chat completions、流式、usage；协议层交给 deepseek-recipe |
| NVLink Engram fetch（#99） | `engram_nvlink.py`，无 GIN 时的取数路径 |
| **本仓库的改动** | Engram fetch 去冗余（性能 2×）、`--cp` 透传、图片输入、模型名可配 |

## 分项补丁

`pr144` / `pr147` / `pr195` / `pr197` / `pr99` 是上游各 PR 的原始补丁，
仅作阅读参考，单独应用可能与 `deepinfer-h20-full.patch` 冲突。
