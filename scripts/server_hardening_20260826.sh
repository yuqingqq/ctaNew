#!/usr/bin/env bash
# R-153 residual hardening (run once as root): systemd-oomd + 4G swapfile +
# conservative swappiness. Idempotent. Context: COORDINATION.md R-148..R-153.
set -euo pipefail
echo "== 1/3 systemd-oomd =="
systemctl enable --now systemd-oomd
systemctl is-active systemd-oomd
echo "== 2/3 4G swapfile (emergency reclaim room; ends livelocks in kills) =="
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
swapon --show
echo "== 3/3 swappiness=10 (swap is emergency-only; protects solver latency) =="
sysctl -w vm.swappiness=10
grep -q '^vm.swappiness' /etc/sysctl.d/99-research-guard.conf 2>/dev/null || echo 'vm.swappiness=10' > /etc/sysctl.d/99-research-guard.conf
echo "== DONE. Verify oomd sees the research slice: =="
oomctl | head -20 || true
