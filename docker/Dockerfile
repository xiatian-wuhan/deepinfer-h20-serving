# DeepSeek-V4.1-Flash-1017 on 8x H20-3e (SM90), one image: engine + OpenAI + Anthropic.
#
# Source: repos/deepinfer-sm90-92-p195-p147 @ 885eea2, which stacks
#   #92  DarkraiHL SM90 (Hopper) port        — the base everything else applies onto
#   #144 host prefix cache (deepinfer/hicache, Sequence.block_table)
#   #195 Scheduler::free_finished_seqs / add_prompted_sequence
#   #147 serve/ OpenAI-compatible layer
# plus yiliu30/deepinfer-server (Anthropic Messages) in /opt/deepinfer/deepinfer-server.
#
# One pass through the project's own ./build.sh, so the vendor wheels and deepinfer._C both
# come from the same commit:
#   pip install -r requirements-nvidia.txt    torch==2.13.0+cu132 + flash-attn-4 + dev
#   pip install -r requirements-sm90.txt      humming-kernels (SM90 routed-expert backend)
#   pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3
#   bash scripts/build_vendor.sh              DeepSelect / FlashMLA-SM90 / DeepEP / DeepGEMM
#   bash scripts/build.sh build               deepinfer._C
# Never hand-assemble those pip calls: pinning nccl in the same resolver pass as torch makes
# pip backtrack over 554 MB torch candidates (torch 2.14.1 -> 2.14.0 -> ...), it never finishes.
FROM lmsysorg/sglang:v0.5.22

# Do NOT put /usr/local/cuda-13.0/compat on LD_LIBRARY_PATH: it contains a 580.126.20 forward-compat
# libcuda.so that shadows the real driver library (this host runs 595.91.07 / CUDA 13.2), which makes
# torch fail with "Error 803: system has unsupported display driver / cuda driver combination".
ENV DEBIAN_FRONTEND=noninteractive \
    DEEPINFER_ACCELERATOR=nvidia \
    DEEPINFER_CUDA_ARCH=sm90 \
    TORCH_CUDA_ARCH_LIST=9.0a \
    MAX_JOBS=16 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

ENV PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple \
    PIP_TRUSTED_HOST=repo.huaweicloud.com \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180

# doc/sm90.md: the DeepGEMM and DeepEP builds need libdw-dev (elfutils headers); the base image
# ships only the libdw1t64 runtime.
# pciutils: vendor/DeepSelect/setup.py imports tests.kernelkit at module scope, which shells out
# to `lspci` before DEEP_SELECT_BUILD_TARGET_PLATFORM is consulted -- without it the build dies
# with FileNotFoundError.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libdw-dev pciutils \
 && rm -rf /var/lib/apt/lists/*

# Isolated venv. setup.py hard-requires CONDA_PREFIX or VIRTUAL_ENV, and putting the venv on
# PATH does NOT set VIRTUAL_ENV -- it has to be declared explicitly.
RUN python -m venv /opt/deepinfer/venv
ENV PATH=/opt/deepinfer/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/deepinfer/venv
RUN pip install -q --upgrade pip wheel setuptools

# github.com is ~20 s to connect from this host and git-over-https rarely completes; gh-proxy
# answers in ~2 s. The submodules are also pre-initialized in the build context, so this is a
# safety net only.
RUN git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/" \
 && git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "git@github.com:"

WORKDIR /opt/deepinfer
COPY repos/deepinfer-sm90-92-p195-p147 /opt/deepinfer/DeepInfer
WORKDIR /opt/deepinfer/DeepInfer

# build_vendor.sh runs `git submodule update --init` over *every* registered submodule, including
# vendor/DeepEP-Ascend and vendor/DeepGEMM-Ascend, which are Ascend-only and useless for the
# nvidia/sm90 target. Measured 2026-10-22: setting submodule.<name>.active=false (below) does NOT
# stop that command from cloning them -- they were still fetched (~1.5 s each, through the global
# gh-proxy insteadOf above). So treat these two lines as decoration, not a guard: a build needs
# github/gh-proxy reachability at this step, and a one-off outage here fails the build with a
# 135 s connect timeout. Retrying the build is the fix; the vendored kernels for sm90 do not
# depend on these two repos at all.
RUN git config submodule.vendor/DeepEP-Ascend.active false \
 && git config submodule.vendor/DeepGEMM-Ascend.active false

RUN --mount=type=cache,target=/root/.cache/pip bash ./build.sh

# rdma-core >= 54. When NCCL probes the DMA-BUF route for GPUDirect RDMA it does
#   dlvsym(libmlx5, "mlx5dv_reg_dmabuf_mr", "MLX5_1.25")
# The base image ships rdma-core 50.0 (libmlx5 ABI 1.24), which does not define that symbol at
# all, so NCCL logs "dlvsym failed on mlx5dv_reg_dmabuf_mr ... version MLX5_1.25" and DeepEP's
# Engram buffer can never initialise -- DeepSeek-V4.1's n-gram tables live behind it.
# rdma-core 54 is the release that introduced the symbol
# (debian/ibverbs-providers.symbols: "mlx5dv_reg_dmabuf_mr@MLX5_1.25 54"); it installs
# libmlx5.so.1.25.54.0 over the distro copy, same soname, so nothing else has to change.
# The final nm/grep makes the build fail loudly if a future rdma-core drops the symbol.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        cmake make g++ pkg-config libnl-3-dev libnl-route-3-dev libudev-dev python3 \
        curl ca-certificates \
 && curl -fsSL -o /tmp/rdma-core.tar.gz \
        https://gh-proxy.com/https://github.com/linux-rdma/rdma-core/archive/refs/tags/v54.0.tar.gz \
 && mkdir -p /tmp/rdma-core \
 && tar xzf /tmp/rdma-core.tar.gz -C /tmp/rdma-core --strip-components=1 \
 && cmake -S /tmp/rdma-core -B /tmp/rdma-core/build \
        -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
        -DNO_MAN_PAGES=1 -DNO_PYVERBS=1 -DENABLE_RESOLVE_NEIGH=0 \
 && cmake --build /tmp/rdma-core/build -j"$(nproc)" \
 && cmake --install /tmp/rdma-core/build \
 && ldconfig \
 && nm -D --defined-only /usr/lib/x86_64-linux-gnu/libmlx5.so.1 | grep -q mlx5dv_reg_dmabuf_mr \
 && readlink -f /usr/lib/x86_64-linux-gnu/libmlx5.so.1 \
 && rm -rf /tmp/rdma-core /tmp/rdma-core.tar.gz /var/lib/apt/lists/*

# The copied tree also carries DeepInfer PR#99 (branch `pr99-engram-nvlink`, commit 63b430a):
# a pure-NVLink Engram fetch for single-node runs without NCCL GIN. DeepSeek-V4.1's n-gram tables
# are fetched with `comm::NCCLGin` even when all ranks are local, so without this patch the model
# cannot start on a host whose driver gives no GPUDirect RDMA (which is the case here: driver
# 595.91.07 ships a vestigial nvidia-peermem and reports DMA_BUF_SUPPORTED=0). See notes/09-*.
# The patch is *in the tree*, not applied here, so assert it is actually present: a stale
# checkout would otherwise build an image that fails only at warmup, long after the build.
RUN grep -q NVLinkEngramBuffer /opt/deepinfer/DeepInfer/deepinfer/model_executor/parallel_utils/userbuffer.py \
 && test -f /opt/deepinfer/DeepInfer/deepinfer/model_executor/parallel_utils/engram_nvlink.py \
 && echo "PR#99 NVLink Engram fetch present"

# Anthropic Messages server. deepseek-recipe is a Rust extension but ships as a
# cp310-abi3 manylinux_2_28_x86_64 wheel, so no Rust toolchain is needed.
COPY repos/deepinfer-server /opt/deepinfer/deepinfer-server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /opt/deepinfer/deepinfer-server/requirements.txt

ENV DEEPINFER_REPO=/opt/deepinfer/DeepInfer \
    DEEPINFER_ENV=/opt/deepinfer/venv/bin/activate \
    PYTHONPATH=/opt/deepinfer/DeepInfer:/opt/deepinfer/deepinfer-server

# Real self-check: no "|| true", and torch is imported first because deepinfer._C links
# libc10.so / libtorch.so, which only enter the loader search path once torch is initialized.
COPY docker/verify_service.py /tmp/verify_service.py
# The build container has no driver, so libcuda.so.1 exists only as the CUDA 13.0
# forward-compat stub. Point the loader at it FOR THIS STEP ONLY: putting it in the image
# ENV would shadow the real driver library at runtime and torch would fail with
# "Error 803: system has unsupported display driver / cuda driver combination".
RUN LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/cuda/lib64 python /tmp/verify_service.py

# OpenAI layer : serve/server.sh start --dp <n> --port 8888   (needs DEEPINFER_HF_MODEL_DIR)
# Anthropic    : deepinfer-server/start.sh                    (needs GATEWAY_MODEL, GATEWAY_GPUS)
WORKDIR /opt/deepinfer/DeepInfer
CMD ["/bin/bash"]
