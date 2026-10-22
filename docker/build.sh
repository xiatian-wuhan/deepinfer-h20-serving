#!/bin/bash
# 用法： ./build.sh sglbase|cuda132
set -euo pipefail
ROOT=/home/ubuntu/data1/xia/deepinfer-h20
VARIANT=${1:-sglbase}
TAG=deepinfer-sm90:${VARIANT}-$(date +%Y%m%d)
case "$VARIANT" in
  sglbase) DF=Dockerfile.sglbase ;;
  cuda132) DF=Dockerfile.cuda132 ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
echo "构建 $TAG  使用 $DF"
docker build -f "$ROOT/docker/$DF" -t "$TAG" "$ROOT" 2>&1 | tee "$ROOT/logs/build-${VARIANT}-$(date +%Y%m%d-%H%M).log"
echo "完成： docker images | grep deepinfer"
