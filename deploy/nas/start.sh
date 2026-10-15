#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
node=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)
test -c "$node" && test -c /dev/kfd || { echo "AMD render node or KFD is missing" >&2; exit 1; }
test "$(cat /sys/bus/pci/devices/0000:03:00.0/vendor)" = 0x1002
test "$(basename -- "$(readlink -f /sys/bus/pci/devices/0000:03:00.0/driver)")" = amdgpu
if grep -q '^AMD_RENDER_DEVICE=' .env; then
    sed -i "s|^AMD_RENDER_DEVICE=.*|AMD_RENDER_DEVICE=$node|" .env
else
    printf '\nAMD_RENDER_DEVICE=%s\n' "$node" >> .env
fi
docker compose up -d --no-build
