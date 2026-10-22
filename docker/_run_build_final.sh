#!/bin/bash
set -uo pipefail
cd /home/ubuntu/data1/xia/deepinfer-h20
TS=$(date +%Y%m%d-%H%M)
TAG="deepinfer-sm90:v41-sm90-${TS}"
LOG="logs/build-final-${TS}.log"
echo "$TAG" > logs/last_final_tag.txt
{
  echo "=== BUILD START $(date -Is) tag=$TAG ==="
  docker build --progress=plain -f docker/Dockerfile -t "$TAG" -t deepinfer-sm90:v41-sm90 . 2>&1
  RC=$?
  echo "=== BUILD_EXIT=$RC $(date -Is) ==="
  exit $RC
} > "$LOG" 2>&1
