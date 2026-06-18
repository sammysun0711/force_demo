# hipBLASLt offline GEMM tuning demo

This folder ships **documentation** and a **Jupyter walkthrough** for AMD **hipBLASLt** offline GEMM tuning (QuickTune: capture hipblaslt-bench lines → tune → analyze → apply `tuning.txt` at inference).

| File | Role |
|------|------|
| [`prepare_env.sh`](prepare_env.sh) | Shell helper for **Step 0** in the markdown: `ROCM_BRANCH` / `GPU_ARCH` / `REPO_ROOT`, clone `rocm-libraries`, `apt` GTest packages, `./install.sh` for hipBLASLt clients. |
| [`launch_hipblaslt_demo.sh`](launch_hipblaslt_demo.sh) | Starts Jupyter Notebook on **`0.0.0.0`** (default port **8888**) for **`hipblaslt_offline_tuning.ipynb`**; override with **`NOTEBOOK_PORT`**. |
| [`hipblaslt_offline_tuning.md`](hipblaslt_offline_tuning.md) | Full command-line flow, `REPO_ROOT` / `QUICKTUNE` layout, troubleshooting, references. |
| [`hipblaslt_offline_tuning.ipynb`](hipblaslt_offline_tuning.ipynb) | Same pipeline in cells: path setup, optional log capture, `gemm_tuning.py` / `tuning_analysis.py` via `subprocess`, CSV preview, override env, cleanup. |

For the upstream library overview, see [What is hipBLASLt?](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/what-is-hipBLASLt.html).

## Prerequisites

- **ROCm** and a GPU that matches your build flags (`HIP_VISIBLE_DEVICES` as needed).
- **`hipblaslt-bench`** on `PATH` after **Step 0** — run [`prepare_env.sh`](prepare_env.sh) from this directory (see [`hipblaslt_offline_tuning.md`](hipblaslt_offline_tuning.md) § Step 0). Typical install output:  
  `…/hipblaslt/build/release/clients`.
- **GTest** (`libgtest-dev`, `libgmock-dev`) if CMake enables tests — `prepare_env.sh` installs them, or use **`BUILD_TESTING=OFF`** in your own configure flags.
- **Python (notebook):** `pandas` (optional CSV tables) and **`ipywidgets`** (dropdowns for **RUN_GEMM_TUNING** and **RUN_CLEANUP_OFFLINE_RESULT**). The notebook starts with `%pip install "pandas" "ipywidgets"` for the active kernel.

If **`gemm_tuning.py`** fails inside `parse_hipblaslt_output`, run the first line of `baseline_reproduce_commands.log` by hand and compare **stdout vs stderr** from `hipblaslt-bench` (see QuickTune README / issues).

## Running the notebook

1. From this directory, **`./launch_hipblaslt_demo.sh`** starts Jupyter on **`http://0.0.0.0:8888/`** with **`hipblaslt_offline_tuning.ipynb`** (requires `pip install notebook` or a conda env with Jupyter). Use **`NOTEBOOK_PORT=9999 ./launch_hipblaslt_demo.sh`** if **8888** is busy. The launcher disables token/password for convenience; do not expose that port on untrusted networks. Alternatively, open the notebook in any Jupyter environment whose **kernel cwd** satisfies `find_repo_roots()` (see above).
2. Run cells top to bottom at least once for imports and `REPO_ROOT` / `QUICKTUNE` / `OUTPUT_PATH`.
3. **Step 2 (GEMM tuning)** and **Cleanup** use **ipywidgets** `Dropdown` widgets (`False` / `True`). The widget is **created once per kernel**; after you change the value, **re-run that same code cell** so Python reads the updated `.value` before `subprocess.run` or `shutil.rmtree` runs.
4. Long GPU work stays in the notebook via **`subprocess`**; keep `capture_output=False` (or equivalent) if you want live bench output in the UI.

## Pipeline (high level)

1. **Build (optional)** — `install.sh` / clients so **`hipblaslt-bench`** exists; prepend `build/release/clients` to `PATH`.
2. **Capture** — `HIPBLASLT_LOG_MASK=32` and `HIPBLASLT_LOG_FILE=…` during inference (or use QuickTune’s bundled `example/Qwen3-32B_hipblaslt.log`).
3. **Tune** — from `utilities/QuickTune`, `gemm_tuning.py` with `--input_file`, `--output_path offline_tuning_result`, `--requested_solution`, `--swizzleA` (see markdown for exact flags).
4. **Analyze** — `tuning_analysis.py` on `unique_*.log` + `tuning_result.csv` → `analysis.csv`.
5. **Apply** — `HIPBLASLT_TUNING_OVERRIDE_FILE` pointing at `offline_tuning_result/tuning.txt` (and unset conflicting `HIPBLASLT_TUNING_FILE` if appropriate). With **`--swizzleA`**, bench commands may use **`--transA T`**; your deployment must match that layout (weights / preshuffle).
6. **Optional** — load `tuning_result.csv` / `analysis.csv` with pandas.
7. **Cleanup** — remove `offline_tuning_result` for a clean next run (destructive; guarded path checks in the notebook).

Artifacts under **`offline_tuning_result/`** typically include: `unique_*.log`, `tuning.txt`, `tuning_result.csv`, `baseline_reproduce_commands.log`, `tuning_reproduce_commands.log`.

## References

- AMD walkthrough: [hipBLASLt offline tuning (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt_offline_tuning/README.html)  
- In-tree QuickTune: `rocm-libraries/projects/hipblaslt/utilities/QuickTune/README.md`
