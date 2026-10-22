#!/bin/bash
set -uo pipefail
REPO=/home/ubuntu/data1/xia/deepinfer-h20/repos/deepinfer-sm90-92-p144
cd "$REPO"

# gh-proxy: 直连 github 建连需 ~20s 且 git 协议几乎不可用；代理秒回
GIT_REWRITE=(
  -c 'url.https://gh-proxy.com/https://github.com/.insteadOf=https://github.com/'
  -c 'url.https://gh-proxy.com/https://github.com/.insteadOf=git@github.com:'
)

echo "=== [$(date +%H:%M:%S)] 标记 Ascend 子模块为 inactive（容器内不 clone）==="
git config submodule.vendor/DeepEP-Ascend.active false
git config submodule.vendor/DeepGEMM-Ascend.active false
echo "  done"

for p in DeepSelect FlashMLA-SM90 DeepEP DeepGEMM; do
  echo "=== [$(date +%H:%M:%S)] init $p ==="
  git -C "vendor/$p" "${GIT_REWRITE[@]}" submodule update --init --recursive 2>&1 | tail -6
  echo "  rc=$?"
  echo "  status:"
  git -C "vendor/$p" submodule status 2>&1 | sed 's/^/    /'
done
echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
