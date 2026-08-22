#!/bin/bash
echo "=== 系统运行时间 ==="
uptime
echo "=== 最近重启记录 ==="
last reboot 2>/dev/null | head -3 || echo "last 不可用"
echo "=== 自动关机配置 ==="
ls /root/autodl-tmp/*.sh 2>/dev/null; cat /etc/crontab 2>/dev/null | grep -iE "shutdown|poweroff|autodl" | head -3
crontab -l 2>/dev/null | grep -iE "shutdown|poweroff" | head -3
echo "=== AutoDL 关机脚本 ==="
ls /root/*.sh 2>/dev/null | head -5
grep -r "shutdown" /root/*.sh /etc/cron* 2>/dev/null | head -5
