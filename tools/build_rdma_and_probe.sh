#!/bin/bash
# 编译最新 rdma-core 覆盖系统库，然后立刻跑 GIN 探针，验证 dma-buf 路线是否点亮 GIN。
# 不动镜像、不动宿主机、不影响 glm53-flash（只用 2 张卡 + 小张量）。
set -x
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    cmake make gcc g++ pkg-config curl ca-certificates \
    libnl-3-dev libnl-route-3-dev libudev-dev python3
cd /tmp
curl -fsSL -o rc.tar.gz https://gh-proxy.com/https://github.com/linux-rdma/rdma-core/archive/refs/heads/master.tar.gz || exit 10
tar xzf rc.tar.gz || exit 11
cd rdma-core-master || exit 12
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
      -DNO_MAN_PAGES=1 -DNO_PYVERBS=1 -DENABLE_RESOLVE_NEIGH=0 .. > /out/cmake.log 2>&1 || { tail -20 /out/cmake.log; exit 13; }
make -j"$(nproc)" > /out/make.log 2>&1 || { tail -30 /out/make.log; exit 14; }
make install > /out/install.log 2>&1 || { tail -20 /out/install.log; exit 15; }
ldconfig
echo "=========== RDMA-CORE BUILT ==========="
readlink -f /usr/lib/x86_64-linux-gnu/libmlx5.so.1
echo -n "mlx5dv_reg_dmabuf_mr present: "
nm -D --defined-only /usr/lib/x86_64-linux-gnu/libmlx5.so.1 | grep -c mlx5dv_reg_dmabuf_mr
echo "=========== GIN PROBE ==========="
cd /opt/deepinfer/DeepInfer
timeout 180 python vendor/DeepEP/tests/engram/test_engram.py \
  --num-processes 2 --num-entries-per-layer 1024,1024 --hidden 512 \
  --num-tokens 8 --num-entries-per-token 2 > /out/probe.log 2>&1
echo "probe exit=$?"
echo -n "GDR/GIN warnings: "; grep -c 'Unable to use GIN' /out/probe.log
grep -m2 -E 'Unable to use GIN|Test passed|Assertion|Traceback' /out/probe.log | cut -c1-200
