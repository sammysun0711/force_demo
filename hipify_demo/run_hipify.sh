#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
/opt/rocm/libexec/hipify/hipify-perl depthwise_conv3d.cu -o depthwise_conv3d.hip -print-stats 2>&1 | tee hipify.log