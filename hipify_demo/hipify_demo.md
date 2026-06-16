# Hipify demo: CUDA source → HIP source → AMD GPU binary

## What is HIPIFY

HIPIFY is a ROCm toolchain that helps migrate CUDA-style GPU code to **HIP** for AMD GPUs. It includes two translators at different sophistication levels:

- **hipify-clang:** A Clang-based tool that parses CUDA, rewrites syntax, API calls, and kernel launches, with stronger diagnostics than the Perl wrapper.
- **hipify-perl:** A lighter rewriter (generated from hipify-clang) that mostly substitutes CUDA identifiers with HIP equivalents. It suits small or mechanical ports but catches fewer issues when translation is ambiguous.

Reference: [AMD HIPIFY documentation](https://rocm.docs.amd.com/projects/HIPIFY).

This demo shows how to take a single file, `depthwise_conv3d.cu`, written with **CUDA APIs**, run it through **hipify-perl** to produce `depthwise_conv3d.hip` (**HIP APIs**), then compile that with **hipcc** for AMD GPUs.

A runnable version of this walkthrough is **[`hipify_demo.ipynb`](hipify_demo.ipynb)** (same steps: setup → hipify → build → run).

## Prerequisites

| Step | Requirement |
|------|-------------|
| Hipify + HIP build | ROCm installed; **`hipcc`** on `PATH` (typical: `/opt/rocm/bin`). This demo invokes **`hipify-perl`** from a fixed path inside `run_hipify.sh` (`/opt/rocm/libexec/hipify/hipify-perl`), not via `PATH`. |

From the repository root, the demo directory is:

```text
hipify/
  depthwise_conv3d.cu   # CUDA source (host + device)
  run_hipify.sh         # CUDA → HIP translation
  build_kernel.sh       # Compile depthwise_conv3d.hip → executable
  run_kernel.sh         # Optional: run binary and save log
  hipify_demo.md        # This document
  hipify_demo.ipynb     # Runnable Jupyter walkthrough (same steps)
```

All script paths below assume your **current working directory** is `hipify/`.

```bash
cd hipify
```

## What the CUDA program does

`depthwise_conv3d.cu` implements a **depthwise 3D convolution** benchmark in BF16 precision: allocates device buffers, runs a **portable reference kernel** (grid-stride loop), times it with GPU events, and optionally checks results against a **CPU float32 reference**.

## Step 1 — Translate CUDA to HIP (`run_hipify.sh`)

Invokes ROCm’s Perl-based hipify on the CUDA file and writes `depthwise_conv3d.hip`. Statistics and messages are tee’d to **`hipify.log`** (see `run_hipify.sh`).

In **[`hipify_demo.ipynb`](hipify_demo.ipynb)**, a cell **prints `run_hipify.sh`** (read-only) before the cell that executes it.

```bash
chmod +x run_hipify.sh   # once, if needed
./run_hipify.sh
```

**Check:**
- `depthwise_conv3d.hip` exists.
- `hipify.log` lists rewrites (for example `cudaMalloc` → `hipMalloc`, `cuda_bf16.h` → `hip/hip_bf16.h`).

**Compare `.cu` / `.hip`:** see **[`hipify_demo.ipynb`](hipify_demo.ipynb)** — it prints **`main` (lines 368–380, 1-based)** side by side for `.cu` and `.hip`.

## Step 2 — Compile the HIP source (`build_kernel.sh`)

Compiles the generated HIP file with `hipcc`, C++17, and an AMDGPU `--offload-arch`.

```bash
chmod +x build_kernel.sh
./build_kernel.sh
```

**Architecture selection:**

- If `HIP_ARCH` is **unset** and `rocminfo` is available, the script tries to pick a `gfx*` name from the **first** GPU entry (heuristic; verify on multi-GPU machines).
- Override explicitly when autodetect is wrong or you are cross-compiling:

```bash
HIP_ARCH=gfx942 ./build_kernel.sh
```

**Optional compile-time knobs**:

```bash
KD=3 KH=5 KW=5 PaddingD=0 PaddingH=2 PaddingW=2 BLOCK_H=45 BLOCK_W=80 ./build_kernel.sh
```

**Check:** executable `depthwise_conv3d` is produced in `hipify/`.

## Step 3 — Run the HIP binary

Program arguments: `[niter] [nwarmup] [--no-check]`

```bash
./depthwise_conv3d                     # defaults: 10 timed iters, 10 warmup, check on
./depthwise_conv3d 5 2                 # 5 timed, 2 warmup
./depthwise_conv3d 10 10 --no-check    # skip CPU reference check (faster)
```

**Optional wrapper** (`run_kernel.sh` — run from `hipify/` so `./depthwise_conv3d` resolves):

```bash
chmod +x run_kernel.sh
./run_kernel.sh                 # tees stdout/stderr to run_depthwise_conv3d.log
```

## Jupyter notebook

**[`hipify_demo.ipynb`](hipify_demo.ipynb)** walks through the same steps in order: locate `hipify/`, **display `run_hipify.sh`**, run **`run_hipify.sh`**, print **`hipify.log`**, compare **`.cu` vs `.hip` around `main` (lines 368–380)**, run **`build_kernel.sh`**, then **`./depthwise_conv3d 2 2`**. Set **`HIP_ARCH`** (environment or uncomment in the notebook build cell) if **`hipcc`** ISA autodetect fails.

Keep GPU runs short when sharing a machine; prefer **`subprocess`** with sensible timeouts in automation.

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `depthwise_conv3d.hip not found` | Run `./run_hipify.sh` from `hipify/` first (same directory as `build_kernel.sh`). |
| `hipify-perl` not found / wrong ROCm layout | `run_hipify.sh` calls `/opt/rocm/libexec/hipify/hipify-perl`. Adjust the path or install ROCm so that binary exists. |
| hipcc / ISA errors | Set `HIP_ARCH` explicitly to your GPU. The build script uses `/opt/rocm/bin/rocminfo` when `HIP_ARCH` is unset; you can also run that binary manually to read the `gfx*` name. |
| Empty or wrong device name in printed header | Cosmetic on some HIP builds; timing and correctness still apply. |
