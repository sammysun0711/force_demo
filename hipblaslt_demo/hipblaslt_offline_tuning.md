## Demo 2: hipBLASLt offline tuning

End-to-end flow: **build** hipBLASLt clients → **capture** GEMM lines from inference → **tune** with `gemm_tuning.py` → **analyze** → **apply** `tuning.txt` (or override) at inference. Use this doc as the outline for a Jupyter notebook (one section per major step; long GPU work via `subprocess` with logs displayed in the notebook).

### Prerequisites (read once)

- **ROCm** and a matching **GPU** (`HIP_VISIBLE_DEVICES` as needed).
- **`hipblaslt-bench`** on `PATH` after building clients (Step 0). QuickTune README: `utilities/QuickTune/README.md`.
- **CMake configure** for hipBLASLt may require **GTest** if tests are enabled: `libgtest-dev` / `libgmock-dev` (see Step 0), or configure with **`BUILD_TESTING=OFF`** if you maintain your own CMake flags.
- **`gemm_tuning.py`** expects **`hipblaslt-bench` stdout** in a fixed shape; if tuning crashes inside `utils.parse_hipblaslt_output`, run the first line of `baseline_reproduce_commands.log` manually and check **stdout vs stderr** (see QuickTune troubleshooting in repo issues / prior analysis).

### Step 0: Prepare environment

Adjust **`ROCM_BRANCH`**, **`GPU_ARCH`**, and **`REPO_ROOT`** to your machine.

```bash
export ROCM_BRANCH=release/rocm-rel-7.2
export GPU_ARCH="gfx942"   # e.g. gfx942, gfx90a — match your hardware
export REPO_ROOT="${REPO_ROOT:-$HOME/workspace/bytedance/demo/hipblaslt-tuning}"

git clone https://github.com/ROCm/rocm-libraries -b "${ROCM_BRANCH}" "${REPO_ROOT}/rocm-libraries"
cd "${REPO_ROOT}/rocm-libraries/projects/hipblaslt"

sudo apt-get update && sudo apt-get install -y libgtest-dev libgmock-dev

./install.sh -c -n -a "${GPU_ARCH}" --skip_rocroller
```

After a successful build, add **hipblaslt-bench** to `PATH` (path matches a typical `install.sh` layout):

```bash
export PATH="${PATH}:${REPO_ROOT}/rocm-libraries/projects/hipblaslt/build/release/clients"
```

### Step 1: Collect hipBLASLt log from model inference

`HIPBLASLT_LOG_MASK=32` makes the library emit **hipblaslt-bench–style** command lines suitable for QuickTune. Replace `run_model.py` with your own inference driver if needed.

```bash
export HIPBLASLT_LOG_MASK=32
export HIPBLASLT_LOG_FILE=Qwen3-32B_hipblaslt.log
# export HIP_VISIBLE_DEVICES=0
python3 run_model.py
```

### Step 2: Run GEMM offline tuning

Run from **`utilities/QuickTune`**. For a **notebook**, prefer foreground `python3 ...` (or `subprocess.run` with `capture_output=False`) instead of `nohup` so cells show progress; keep `nohup` for unattended servers.

```bash
export QUICKTUNE="${REPO_ROOT}/rocm-libraries/projects/hipblaslt/utilities/QuickTune"
cd "${QUICKTUNE}"

# Optional: short dry run on a tiny log first to validate PATH and parsing
python3 gemm_tuning.py \
  --input_file example/Qwen3-32B_hipblaslt.log \
  --output_path offline_tuning_result \
  --requested_solution 128 \
  --swizzleA
```

Background variant (servers):

```bash
nohup python3 gemm_tuning.py --input_file example/Qwen3-32B_hipblaslt.log --output_path offline_tuning_result --requested_solution 128 --swizzleA > output.log 2>&1 &
```

**Artifacts** (under `offline_tuning_result/`): `unique_*.log`, `tuning.txt`, `tuning_result.csv`, `baseline_reproduce_commands.log`, `tuning_reproduce_commands.log`.

### Step 3: Analyze GEMM tuning result

Still from **`utilities/QuickTune`** (same shell as Step 2, or `cd "${QUICKTUNE}"`).

```bash
python3 tuning_analysis.py \
  --input_log offline_tuning_result/unique_Qwen3-32B_hipblaslt.log \
  --input_csv offline_tuning_result/tuning_result.csv \
  --output_csv offline_tuning_result/analysis.csv
```

Example **stdout** (shape varies with log):

```text
Total -m -n -k combos: 13

                  (-m, -n, -k)   count  baseline/tuned  total_baseline(us)   total_tuned(us)   baseline%    tuned%
------------------------------------------------------------------------------------------------------------------
            (6400, 4000, 5120)       1       109.22%            1397.890          1279.890       45.42%     48.95%
            (5120, 4000, 3200)       1       123.99%             789.434           636.687       25.65%     24.35%
            (5120, 4000, 1024)       1       131.57%             312.454           237.489       10.15%      9.08%
            (1280, 4000, 5120)       1       105.82%             284.756           269.085        9.25%     10.29%
              (18992, 1, 5120)       1       149.51%              88.168            58.970        2.86%      2.26%
               (6400, 8, 5120)       1       165.01%              43.779            26.531        1.42%      1.01%
               (6400, 1, 5120)       1       167.09%              43.478            26.021        1.41%      1.00%
               (5120, 1, 3200)       1       131.99%              22.955            17.392        0.75%      0.67%
               (5120, 8, 3200)       1       123.52%              22.308            18.061        0.72%      0.69%
               (1280, 8, 5120)       1       179.64%              20.518            11.422        0.67%      0.44%
               (1280, 1, 5120)       1       177.26%              20.318            11.462        0.66%      0.44%
               (5120, 8, 1024)       1       144.38%              15.889            11.005        0.52%      0.42%
               (5120, 1, 1024)       1       142.65%              15.587            10.927        0.51%      0.42%
------------------------------------------------------------------------------------------------------------------
Gemm Speedup: 1.18x
Total baseline time: 3077.534 us, Total tuned time: 2614.941 us

Output CSV: offline_tuning_result/analysis.csv
```

### Step 4: Adopt tuned solution

**Important:** With **`--swizzleA`**, QuickTune adjusts the bench command (e.g. **`--transA T`**). Tuned kernels then assume the **A** matrix layout matches that configuration—your integration may need **weight transpose / preshuffle** after load so calls match the tuned solution (see AMD blog below).

Use **`HIPBLASLT_TUNING_OVERRIDE_FILE`** to apply the generated **`tuning.txt`** without overwriting a development **`HIPBLASLT_TUNING_FILE`**:

```bash
unset HIPBLASLT_TUNING_FILE
export HIPBLASLT_TUNING_OVERRIDE_FILE="${QUICKTUNE}/offline_tuning_result/tuning.txt"
python3 run_model.py
```

### Notebook outline (for a future `hipblaslt_offline_tuning.ipynb`)

1. **Markdown** — goal, prerequisites, pipeline diagram (log → unique → bench → `tuning.txt` → inference).
2. **Code** — `os.environ` / `Path` / `REPO_ROOT`; optional `!` or `subprocess` to show `hipblaslt-bench --version`.
3. **Markdown** — Step 1: log capture; **Code** — stub or `print` instructions if `run_model.py` is not in-repo.
4. **Code** — `cd` QuickTune; run `gemm_tuning.py` with **`subprocess`** (long run: log tail from `output.log`, or stream stdout).
5. **Code** — load **`tuning_result.csv`** / **`analysis.csv`** with **pandas**; small charts (optional).
6. **Markdown** — Step 4 caveats (**swizzleA**, **OVERRIDE_FILE**).

Reference:

- AMD walkthrough: [hipBLASLt offline tuning (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt_offline_tuning/README.html)
- In-tree QuickTune details: `rocm-libraries/projects/hipblaslt/utilities/QuickTune/README.md`
