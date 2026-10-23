#!/bin/bash
# 恢复运维原有的 GRUB_DEFAULT 写法（标题式，声明在 /etc/default/grub 里，
# 比 saved_entry 更持久），只保留 iommu=pt 这一处变化。
set -uo pipefail
sed -i 's|^GRUB_DEFAULT=.*|GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-57-generic"|' /etc/default/grub
echo "=== 当前 /etc/default/grub ==="
grep -E '^GRUB_(DEFAULT|CMDLINE)' /etc/default/grub
echo
echo "=== 与备份的差异（应只有 GRUB_CMDLINE_LINUX 一行）==="
diff /root/grub.bak.2026-10-22-155257 /etc/default/grub
echo "(diff 结束)"
echo
echo "=== update-grub ==="
update-grub 2>&1 | tail -2
echo
echo "=== 确认 grub.cfg 里 iommu=pt 覆盖了所有条目 ==="
grep -c 'iommu=pt' /boot/grub/grub.cfg
