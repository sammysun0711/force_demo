#!/usr/bin/env bash
# Step 0 from hipblaslt_offline_tuning.md: clone rocm-libraries (hipBLASLt), install deps, build clients.
set -euo pipefail

: "${ROCM_BRANCH:=release/rocm-rel-7.2}"
: "${REPO_ROOT:=$HOME/workspace/bytedance/demo/force_demo/hipblaslt_demo}"
# Default GPU ISA for install.sh -a (MI300-class gfx942). Override: GPU_ARCH=gfx90a ./prepare_env.sh
GPU_ARCH="${GPU_ARCH:-gfx942}"

ROCM_LIBRARIES="${REPO_ROOT}/rocm-libraries"
HIPBLASLT_DIR="${ROCM_LIBRARIES}/projects/hipblaslt"

echo "ROCM_BRANCH=${ROCM_BRANCH}"
echo "GPU_ARCH=${GPU_ARCH}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "HIPBLASLT_DIR=${HIPBLASLT_DIR}"
echo

mkdir -p "${REPO_ROOT}"

if [[ ! -d "${ROCM_LIBRARIES}/.git" ]]; then
  echo "Cloning rocm-libraries (branch ${ROCM_BRANCH})..."
  git clone -b "${ROCM_BRANCH}" \
    https://github.com/ROCm/rocm-libraries \
    "${ROCM_LIBRARIES}"
else
  echo "Existing clone at ${ROCM_LIBRARIES} — skipping git clone (checkout ${ROCM_BRANCH} yourself if needed)."
fi

cd "${HIPBLASLT_DIR}"

echo "Installing distro packages for GTest/GMock (CMake tests)..."
sudo apt-get update
sudo apt-get install -y libgtest-dev libgmock-dev

echo "Running hipBLASLt install.sh (this can take a long time)..."
./install.sh -c -n -a "${GPU_ARCH}" --skip_rocroller

echo
echo "Done. Add hipblaslt-bench to PATH for this shell:"
echo "  export PATH=\"\${PATH}:${HIPBLASLT_DIR}/build/release/clients\""
