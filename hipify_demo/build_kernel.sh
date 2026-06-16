#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -f depthwise_conv3d.hip ]]; then
  echo "error: depthwise_conv3d.hip not found. Run ./run_hipify.sh first." >&2
  exit 1
fi
if [[ -z "${HIP_ARCH:-}" ]] && command -v /opt/rocm/bin/rocminfo &>/dev/null; then
  ARCH=$(/opt/rocm/bin/rocminfo 2>/dev/null | grep "Name:" | grep -oE "gfx[[:alnum:]]+" | head -1)
  ARCH="${ARCH:-gfx942}"
else
  ARCH="${HIP_ARCH:-gfx942}"
fi
hipcc -std=c++17 -O3 --offload-arch="${ARCH}" \
  depthwise_conv3d.hip -o depthwise_conv3d \
  -DKD="${KD:-3}" -DKH="${KH:-5}" -DKW="${KW:-5}" \
  -DPaddingD="${PaddingD:-0}" -DPaddingH="${PaddingH:-2}" -DPaddingW="${PaddingW:-2}" \
  -DBLOCK_H="${BLOCK_H:-45}" -DBLOCK_W="${BLOCK_W:-80}"
echo "Built: ${ROOT}/depthwise_conv3d (HIP_ARCH=${ARCH})"
