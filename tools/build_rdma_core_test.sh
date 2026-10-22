#!/bin/bash
# 在一个 --rm 容器里编译 rdma-core v51.0 并覆盖同 soname 的库，用于验证
# mlx5dv_reg_dmabuf_mr@@MLX5_1.25 是否出现、以及 NCCL 的 GIN dma-buf 路径是否点亮。
# 不动镜像、不动宿主机；验证通过后再进 Dockerfile。
set -x
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    cmake make gcc g++ pkg-config curl ca-certificates \
    libnl-3-dev libnl-route-3-dev libudev-dev python3
cd /tmp
curl -fsSL -o rc.tar.gz https://gh-proxy.com/https://github.com/linux-rdma/rdma-core/archive/refs/tags/v51.0.tar.gz || exit 10
tar xzf rc.tar.gz || exit 11
cd rdma-core-51.0 || exit 12
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
      -DNO_MAN_PAGES=1 -DNO_PYVERBS=1 -DENABLE_RESOLVE_NEIGH=0 .. > /out/cmake.log 2>&1 || { tail -20 /out/cmake.log; exit 13; }
make -j"$(nproc)" > /out/make.log 2>&1 || { tail -30 /out/make.log; exit 14; }
make install > /out/install.log 2>&1 || { tail -20 /out/install.log; exit 15; }
ldconfig
echo "=========== RESULT ==========="
ls -l /usr/lib/x86_64-linux-gnu/libmlx5.so.1
echo -n "mlx5dv_reg_dmabuf_mr: "; nm -D --defined-only /usr/lib/x86_64-linux-gnu/libmlx5.so.1 | grep -c reg_dmabuf_mr
echo -n "symbol version:        "; objdump -p /usr/lib/x86_64-linux-gnu/libmlx5.so.1 | grep -oE 'MLX5_1\.(2[5-9]|[3-9][0-9])' | sort -u -V | tr '\n' ' '
echo
