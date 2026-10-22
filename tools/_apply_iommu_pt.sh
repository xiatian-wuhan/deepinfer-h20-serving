#!/bin/bash
# 在宿主机 root 命名空间里执行：备份、写入 iommu=pt、update-grub、验证。
set -uo pipefail
BAK=/root/grub.bak.$(date +%F-%H%M%S)
echo "=== 1) 备份 /etc/default/grub -> $BAK"
cp -a /etc/default/grub "$BAK" || { echo "BACKUP FAILED"; exit 1; }
ls -l "$BAK"; echo "--- 备份内容 ---"; cat "$BAK"

echo; echo "=== 2) 修改前"
grep -E '^GRUB_CMDLINE' /etc/default/grub

echo; echo "=== 3) 写入 iommu=pt"
sed -i 's|^GRUB_CMDLINE_LINUX=.*|GRUB_CMDLINE_LINUX="iommu=pt"|' /etc/default/grub
grep -E '^GRUB_CMDLINE' /etc/default/grub
if ! grep -qx 'GRUB_CMDLINE_LINUX="iommu=pt"' /etc/default/grub; then echo "SED FAILED"; exit 1; fi

echo; echo "=== 4) update-grub"
update-grub 2>&1 | tail -6

echo; echo "=== 5) 验证 /boot/grub/grub.cfg"
grep -c 'iommu=pt' /boot/grub/grub.cfg
echo "--- 将用于启动的 linux 行 ---"
grep -E '^\s+linux\s' /boot/grub/grub.cfg | head -2

echo; echo "BAK_FILE=$BAK"
