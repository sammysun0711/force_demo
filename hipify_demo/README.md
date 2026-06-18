# HIPIFY demo (CUDA → HIP → AMD GPU)

Small walkthrough that takes **`depthwise_conv3d.cu`** (CUDA), translates it with **`hipify-perl`** to **`depthwise_conv3d.hip`** (HIP), then builds a **`hipcc`** binary you can run on AMD GPUs.

| Resource | Description |
|----------|-------------|
| [`hipify_demo.md`](hipify_demo.md) | Full narrative: HIPIFY background, prerequisites, each step, troubleshooting table. |
| [`hipify_demo.ipynb`](hipify_demo.ipynb) | Same flow in Jupyter: show `run_hipify.sh`, run it, inspect `hipify.log`, diff **`.cu` / `.hip`** around `main`, build, short run. |
| [`launch_hipify_demo.sh`](launch_hipify_demo.sh) | Starts Jupyter Notebook on **`0.0.0.0`** (default port **8888**) so you can open the notebook in a browser; override with **`NOTEBOOK_PORT`**. |
| [`run_hipify.sh`](run_hipify.sh) | Invokes `/opt/rocm/libexec/hipify/hipify-perl` → `depthwise_conv3d.hip`, logs to **`hipify.log`**. |
| [`build_kernel.sh`](build_kernel.sh) | Compiles **`depthwise_conv3d.hip`** with **`hipcc`** → executable **`depthwise_conv3d`**. |
| [`run_kernel.sh`](run_kernel.sh) | Optional: runs the HIP binary and tees **`run_depthwise_conv3d.log`**. |
| [`build_cuda.sh`](build_cuda.sh) | Optional: builds the original **CUDA** variant with **`nvcc`** → **`depthwise_conv3d_cuda`** (NVIDIA path, not required for the HIP demo). |

Upstream docs: [AMD HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY).

## Prerequisites

- **ROCm** with **`hipcc`** on `PATH` (typical: `/opt/rocm/bin`).
- **`hipify-perl`** at the path used in `run_hipify.sh` (default **`/opt/rocm/libexec/hipify/hipify-perl`**). Adjust the script if your install layout differs.

## Quick start

Run everything from this directory:

```bash
cd hipify
chmod +x run_hipify.sh build_kernel.sh run_kernel.sh launch_hipify_demo.sh   # once, if needed
./run_hipify.sh          # produces depthwise_conv3d.hip + hipify.log
./build_kernel.sh        # produces ./depthwise_conv3d (HIP_ARCH autodetect or override)
./depthwise_conv3d 2 2   # example: 2 timed iters, 2 warmup (see hipify_demo.md for flags)
```

**GPU ISA:** if autodetect is wrong or you cross-compile, set **`HIP_ARCH`** (e.g. `HIP_ARCH=gfx942 ./build_kernel.sh`). Optional kernel shape overrides match `build_kernel.sh` / the markdown (`KD`, `KH`, `KW`, padding, block sizes).

**Jupyter:** from this directory, `./launch_hipify_demo.sh` (requires `pip install notebook` or a conda env with Jupyter). Use `NOTEBOOK_PORT=9999 ./launch_hipify_demo.sh` if **8888** is busy; open the URL with token printed in the terminal.

## What the program does

**`depthwise_conv3d.cu`** is a **depthwise 3D convolution** micro-benchmark in **BF16**: device allocation, a portable grid-stride kernel, event timing, and an optional **CPU float32** check. After hipify + build, the HIP binary exposes the same style of CLI: **`[niter] [nwarmup] [--no-check]`**.