#!/bin/bash
ROOT=/home/ubuntu/data1/xia/deepinfer-h20
R=$ROOT/repos
LOG=$ROOT/logs/clone.log
exec >> "$LOG" 2>&1
echo "=================== START $(date -Is) ==================="
clone() {  # name url branch depth
  local name=$1 url=$2 br=$3 depth=$4
  if [ -d "$R/$name/.git" ]; then echo "[skip] $name 已存在"; return; fi
  echo "--- clone $name ($br, depth=$depth)"
  git clone --depth "$depth" --single-branch --branch "$br" --no-tags "$url" "$R/$name" \
    && echo "[ok] $name  $(git -C "$R/$name" rev-parse HEAD)" \
    || echo "[FAIL] $name"
}
clone deepinfer-server    https://github.com/yiliu30/deepinfer-server.git      main          1
clone deepinfer-serving   https://github.com/carlulsoe/deepinfer-serving.git   main          1
clone deepseek-recipe     https://github.com/deepseek-ai/deepseek-recipe.git   main          1
clone deepinfer-main      https://github.com/deepseek-ai/DeepInfer.git         main          1
clone deepinfer-sm90-92   https://github.com/DarkraiHL/DeepInfer.git          sm90-hopper  40
echo "=================== DONE $(date -Is) ==================="
