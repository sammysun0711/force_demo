# Lance × vLLM-Omni demo

![Lance × vLLM-Omni Gradio unified demo UI](demo_example.png)

This demo provide a interactive an interactive **Gradio** UI that runs all **Lance** multimodal tasks in one place.

| Path | Role |
|------|------|
| [`prepare_env.sh`](prepare_env.sh) | Clones [`vllm-omni`](https://github.com/sammysun0711/vllm-omni) on branch **`lance_demo`** into `./vllm-omni` (idempotent if you remove the clone first). |
| [`vllm-omni/examples/offline_inference/lance/gradio_demo.py`](vllm-omni/examples/offline_inference/lance/gradio_demo.py) | Unified Gradio app: text/image/video generation, editing, and video/image understanding. |

Upstream references: [Lance](https://github.com/bytedance/Lance) · [vLLM-Omni](https://github.com/vllm-project/vllm-omni) · [Lance on Hugging Face](https://huggingface.co/bytedance-research/Lance)

## Prerequisites

- **AMD GPU(s)** suitable for Lance inference (the demo defaults to **two logical GPUs**: image Omni + video Omni when `ulysses_degree=1` and `replicas_per_omni=1`).
- **Python environment** with this `vllm-omni`, `Gradio`, `FlashAttention`, etc.
- **Model snapshot** `--model` pointing at a **parent directory** that contains both:

  - `Lance_3B/` — image weights (t2i, image edit, image understanding)
  - `Lance_3B_Video/` — video weights (t2v, i2v, video edit, video understanding)

## Setup

1. From this directory, run:

   ```bash
   bash prepare_env.sh
   ```

2. **Optional — curated local assets** (relative to `gradio_demo.py`):

   | Directory | Task | Contents |
   |-----------|------|----------|
   | `vllm-omni/examples/offline_inference/lance/assets/image_to_video/` | Image → Video | First-frame `.webp` files paired in code with long motion prompts (`00001.webp`, …). |
   | `vllm-omni/examples/offline_inference/lance/assets/video_qa/` | Video understanding | `.mp4` clips paired with VQA / caption prompts (`vqa-001-opt.mp4`, …). |

   Missing files are skipped; existing rows still load.

## Run the Gradio demo

From the **`vllm-omni`** repo root (so imports resolve), with GPUs visible:

```bash
cd vllm-omni
export CUDA_VISIBLE_DEVICES=0,1   # at least 2 devices for default layout
python examples/offline_inference/lance/gradio_demo.py \
  --model /path/to/parent/dir/containing/Lance_3B_and_Lance_3B_Video \
  --host 0.0.0.0 \
  --port 7860
```

Useful flags (see `--help` on the script for the full list):

| Flag | Purpose |
|------|---------|
| `--replicas-per-omni N` | Scale throughput with **N** replicas per Omni (**2×N** GPUs total with default SP layout). |
| `--ulysses-degree K` | Ulysses sequence-parallel degree per Omni (**2×K** GPUs); **i2v is experimental** when `K>1` — prefer replicas for throughput. |
| `--share` | Gradio public share link. |

The UI exposes seven tasks (radio): video generation, video edit, **video understanding**, image-to-video, image generation, image edit, **image understanding**. Text-only presets (t2i / t2v) use full-width clickable blocks; other tasks use `gr.Examples` where media paths must sit under **`allowed_paths`** (the demo adds `--examples-root`, curated `assets/image_to_video`, `assets/video_qa`, and `/tmp/lance_demo_sources` when present).

## Layout summary

```text
lance_demo/
  README.md                 ← this file
  prepare_env.sh            ← clone vllm-omni (branch lance_demo)
  vllm-omni/
    examples/offline_inference/lance/
      gradio_demo.py        ← main UI
      assets/
        image_to_video/     ← optional i2v first frames
        video_qa/           ← optional video-QA clips
      end2end.py            ← other offline Lance entrypoints
    …
```

## Troubleshooting

- **`Lance_3B` / `Lance_3B_Video` not found:** `--model` must be the **parent** of those two directories, not one of the leaf checkpoint folders.
- **Examples do not load media:** ensure files exist under the `assets/…` dirs above or under `--examples-root`, and restart the app after copying assets.
- **OOM or slow first run:** reduce `--replicas-per-omni`, use shorter duration in the UI for video tasks, or scale GPUs per the script’s layout rules.