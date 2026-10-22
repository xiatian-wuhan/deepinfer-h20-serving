#!/bin/bash
# 重启后的一条命令验收脚本。四步，给出 PASS/FAIL 结论。
#   bash tools/post_reboot_check.sh
set -uo pipefail
ROOT=/home/ubuntu/data1/xia/deepinfer-h20
IMAGE=deepinfer-sm90:v41-sm90
echo "=========== h20 GPUDirect RDMA 验收  $(date '+%F %T') ==========="

echo
echo "--- [1/4] 内核命令行是否带 iommu=pt ---"
if tr ' ' '\n' < /proc/cmdline | grep -qx 'iommu=pt'; then
    echo "PASS  iommu=pt 已生效"
else
    echo "FAIL  /proc/cmdline 里没有 iommu=pt（GRUB 没改成功或没 update-grub）"
fi
echo "      cmdline: $(cat /proc/cmdline)"

echo
echo "--- [2/4] IOMMU 默认域类型 ---"
dt=$(docker run --rm --privileged --entrypoint bash "$IMAGE" \
     -lc "dmesg 2>/dev/null | grep -i 'Default domain type' | tail -1" 2>/dev/null)
echo "      ${dt:-（没读到）}"
case "$dt" in
    *Pass-through*|*Passthrough*|*pass-through*) echo "PASS  IOMMU 已是 passthrough" ;;
    *Translated*)                                echo "FAIL  仍是 Translated" ;;
    *)                                           echo "WARN  读不到这一行，以第 3 步为准" ;;
esac

echo
echo "--- [3/4] CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED (attr 124) ---"
out=$(docker run --rm --gpus '"device=0"' -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 "$IMAGE" \
      python -c "
import ctypes
l = ctypes.CDLL('libcuda.so.1'); l.cuInit(0)
d = ctypes.c_int(); l.cuDeviceGet(ctypes.byref(d), 0)
v = ctypes.c_int(); l.cuDeviceGetAttribute(ctypes.byref(v), 124, d)
print(v.value)" 2>&1 | tail -1)
echo "      attr 124 = $out"
if [ "$out" = "1" ]; then echo "PASS  dma-buf 已支持"; else echo "FAIL  dma-buf 仍不支持，GIN 不会通"; fi

echo
echo "--- [4/4] GIN 探针（9 秒，需要 GPU 空闲） ---"
bash "$ROOT/tools/verify_gin.sh" 2>&1 | sed -n '/--- verdict ---/,+3p'

echo
echo "=========== 结束：1-3 全 PASS 且第 4 步 GIN OK，才算通 ==========="
