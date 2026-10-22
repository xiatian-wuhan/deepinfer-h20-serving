> **注意**：本文记录的 `p144-v2-20261022-1202` 是两步构建的**中间产物，已删除**。
> 最终交付是单镜像 `deepinfer-sm90:v41-sm90`（46.2 GB），见 `04-final-image.md`。
> 下文记录的六个坑对单镜像构建依然全部适用（`docker/Dockerfile` 已内含全部修法）。

# 镜像构建记录：deepinfer-sm90:p144-v2-20261022-1202

## 结论

| 项目 | 值 |
| --- | --- |
| 镜像 tag | `deepinfer-sm90:p144-v2-20261022-1202` |
| 镜像 ID / 大小 | `e461cfb30bd3` / **46.1 GB** |
| 基础镜像 | `lmsysorg/sglang:v0.5.22`（CUDA 13.0.88, Python 3.12.15, gcc 13.3） |
| 源码 | `repos/deepinfer-sm90-92-p144`（#92 `e205d42` + #144 → `6985279`） |
| 构建入口 | **`./build.sh`**（官方唯一支持顺序），`DEEPINFER_ACCELERATOR=nvidia` |
| 有效编译耗时 | **425 s**（依赖 pip cache mount 全命中 + 562 个 kernel） |
| GPU 使用 | **0** —— 全程未占用 GPU，生产 `glm53-flash` 无影响 |

## 镜像内实测

```
python            3.12.15
torch             2.13.0+cu132   (torch.version.cuda = 13.2)
deepinfer._C      OK
deep_select       1.0.0            deep_gemm        2.8.2
flash_mla_sm90    1.0.0            deep_ep          2.5.0
humming (pip: humming-kernels)  0.1.12
flash_attn (pip: flash-attn-4)  4.0.0b34,  flash_attn.cute.interface 可用
Sequence.block_table             True    ← #144 生效
```

wheel 版本号带源码提交，可交叉核对 pin：
`deep_gemm-2.8.2+1fc9c66`、`flash_mla_sm90-1.0.0+ba89a34.sm90v41`、`deep_select-1.0.0+bfa4507`。

## 六个导致失败的坑（及修法）

### 1. 手工拼 pip 命令引发 torch 无限回溯（最严重）
把 `nvidia-nccl-cu13==2.32.3` 和 `torch==2.13.0+cu132` 放进**同一次** `pip install`：
torch 硬锁 `nccl==2.29.7`，我们又要 `==2.32.3` → pip 判定 torch 不可用 → 去试
`torch 2.14.1`（554 MB）→ 再冲突 → `2.14.0`（554 MB）→ …，每轮 2 分 18 秒，跑不完。

**修法**：改用官方 `./build.sh`。它第 23 行本来就有注释
`# Patch NCCL to 2.32.3 (after all other pip installs, which may reinstall torch's pinned NCCL)`
并用 `pip install --no-deps --force-reinstall` 在最后单独打补丁——项目作者早就处理了。
另一条 DeepEP README 也写着 `pip install "nvidia-nccl-cu13>=2.32.3" --no-deps`。

### 2. `VIRTUAL_ENV` 未设置 → setup.py 必然崩
`setup.py:56-58` 要求 `CONDA_PREFIX` 或 `VIRTUAL_ENV`，否则 `RuntimeError`。
把 venv 加进 `PATH` **不会**设置该变量。修法：显式 `ENV VIRTUAL_ENV=/opt/deepinfer/venv`。

### 3. 缺 `libdw-dev`
`doc/sm90.md:38`：DeepGEMM 和 DeepEP 编译需要 elfutils 头文件。
基础镜像只有 `libdw1t64`（运行时），无 `libdw.h`。修法：`apt-get install libdw-dev`。

### 4. 缺 `lspci`（pciutils）
`vendor/DeepSelect/setup.py:8` 在 **import 阶段**就执行
`tests/kernelkit/build.py:9 → platform.py:20 subprocess.check_output(["lspci"])`，
镜像无 `pciutils` → `FileNotFoundError: 'lspci'`。
注意 `build_vendor.sh` 会传 `DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA` 覆盖平台判断，
但那个模块级调用发生在覆盖生效之前，所以仍需 `lspci` 存在。
实测容器内 `lspci` 可见 GPU（`3D controller: NVIDIA Corporation Device 232c`），判定为 CUDA。

### 5. 子子模块全未初始化 + GitHub 不可用
`vendor/{DeepSelect,FlashMLA-SM90,DeepGEMM}` 需要 cutlass，`vendor/{DeepEP,DeepGEMM}` 需要 deep_jit，
全部 `-` 未初始化。而容器内直接 clone 不可行：

```
github.com 根                     -> 000 (12s 超时)
github.com/deepseek-ai/DeepEP...  -> 200，但 conn=19.4s / tls=20.3s
git ls-remote                     -> 25s 无输出
```

**修法**：本地经 **gh-proxy.com** 预初始化（`git ls-remote` 2 秒回，cutlass 27-29 秒克隆完），
再用 `docker COPY` 带入；容器内 `git submodule update --init --recursive` 变成 no-op。
另在镜像里设 `git config --global url."https://gh-proxy.com/https://github.com/".insteadOf` 作兜底。
（`ghproxy.net` → 403，`gitclone.com` → 502，`hub.fastgit.org` → 000，只有 gh-proxy 可用。）

### 6. 自检写法错误（我的 bug，非镜像缺陷）
原来写 `python -c "from deepinfer._C import engine"` 单独执行 →
`ImportError: libc10.so: cannot open shared object file`。
原因：`deepinfer._C` 链接 `libc10.so`/`libtorch.so`，它们在 `site-packages/torch/lib`，
**只有 `import torch` 之后才会进入动态链接器搜索路径**。
修法：自检里先 `import torch` 再导入扩展。

## 顺带修正的两处

- `TORCH_CUDA_ARCH_LIST` 从 `9.0` 改为 **`9.0a`**：Hopper 的 wgmma 需要 `sm_90a`，
  DeepSelect 的 `0001-build-sm_90a.patch` 正是打开 `-gencode arch=compute_90a`。
- 删掉原自检的 `|| true`：它让自检永远返回 0，**导入失败也会“构建成功”**，是自欺欺人。

## 未生效的做法（记录以免重试）

- `git config submodule.vendor/DeepEP-Ascend.active false` **不能**阻止
  `git submodule update --init` 克隆它——Ascend 两个子模块在容器内照样被 clone
  （好在很小，共约 3 秒，不影响结果）。

## 未做的验证（需要 GPU，必须在停掉 glm53-flash 之后）

8 张 H20-3e 各被占至 142,870/143,771 MiB（仅剩约 900 MiB），
加载 510 GB 权重需要全部 8 卡，因此：

- `pytest tests/sm90 tests/hicache`
- `examples/offline_inference.py` 冒烟
- `examples/latency_benchmark.py` 复现 TTFT 2.114 s@128K / c64 4259 tok/s
- `deepinfer/hicache/kernels.py` 的 UVA page copy 是**运行时 JIT**
  （`torch.utils.cpp_extension.load`），首次使用会在容器内现场编译，
  需确认 `TORCH_CUDA_ARCH_LIST=9.0a`（已设）与 `~/.cache/torch_extensions` 可写
