#!/usr/bin/env bash
# R-153 residual hardening (run once as root). Idempotent; steps independent —
# a failing step reports and continues. Context: COORDINATION.md R-148..R-153.
# v2: swapfile first; systemd-oomd is a separate package on Ubuntu — install it.
set -u
fail=0

echo "== 1/3 4G swapfile (emergency reclaim room; ends livelocks in kills) =="
if swapon --show | grep -q /swapfile; then
  echo "swapfile already active"
else
  { fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile; } || { echo "SWAPFILE STEP FAILED"; fail=1; }
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
swapon --show || true

echo "== 2/3 swappiness=10 (swap is emergency-only; protects solver latency) =="
sysctl -w vm.swappiness=10 || { echo "SWAPPINESS STEP FAILED"; fail=1; }
echo 'vm.swappiness=10' > /etc/sysctl.d/99-research-guard.conf

echo "== 3/3 systemd-oomd (separate package on Ubuntu; install then enable) =="
if systemctl cat systemd-oomd.service >/dev/null 2>&1; then
  echo "systemd-oomd unit present"
else
  DEBIAN_FRONTEND=noninteractive apt-get install -y systemd-oomd || { echo "OOMD INSTALL FAILED"; fail=1; }
fi
systemctl enable --now systemd-oomd 2>/dev/null || { echo "OOMD ENABLE FAILED"; fail=1; }
systemctl is-active systemd-oomd || true

echo "== summary =="
swapon --show | tail -1 || true
sysctl vm.swappiness || true
command -v oomctl >/dev/null && oomctl | head -15 || echo "(oomctl not available)"
exit $fail
