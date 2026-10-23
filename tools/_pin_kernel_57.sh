#!/bin/bash
# 把 GRUB 默认项钉到 6.8.0-57-generic（当前在跑的、38 天稳定运行的kernel），
# 使本次重启的唯一变量是 iommu=pt，而不是"顺带换了内核"。
set -uo pipefail
ID=$(grep -oE 'gnulinux-6\.8\.0-57-generic-advanced-[0-9a-f-]+' /boot/grub/grub.cfg | head -1)
echo "目标 menuentry id: $ID"
[ -z "$ID" ] && { echo "ID NOT FOUND"; exit 1; }

echo; echo "=== 修改前 ==="
grep -E '^GRUB_(DEFAULT|CMDLINE)' /etc/default/grub

if grep -q '^GRUB_DEFAULT=' /etc/default/grub; then
    sed -i 's|^GRUB_DEFAULT=.*|GRUB_DEFAULT=saved|' /etc/default/grub
else
    echo 'GRUB_DEFAULT=saved' >> /etc/default/grub
fi

echo; echo "=== 修改后 ==="
grep -E '^GRUB_(DEFAULT|CMDLINE)' /etc/default/grub

echo; echo "=== update-grub ==="
update-grub 2>&1 | tail -3

echo; echo "=== grub-set-default ==="
grub-set-default "$ID" && echo "OK"

echo; echo "=== grubenv ==="
grep -o 'saved_entry=.*' /boot/grub/grubenv | head -1
