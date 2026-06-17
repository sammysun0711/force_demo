# ROCm Demo for Force Event

ROCm demo repository for the Force Event, including **ROCm developer tools** and a **Lance** deployment walkthrough:

- [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html)
- [hipBLASLt GEMM tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-offline-tuning.html)
- [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) multimodal model deployment demo.

| Demo | README | Notebook / Gradio |
|------|--------|-------------------|
| **HIPIFY** (CUDA → HIP) | [hipify_demo/README.md](hipify_demo/README.md) | [hipify_demo/hipify_demo.ipynb](hipify_demo/hipify_demo.ipynb) |
| **hipBLASLt** offline GEMM tuning | [hipblaslt_demo/README.md](hipblaslt_demo/README.md) | [hipblaslt_demo/hipblaslt_offline_tuning.ipynb](hipblaslt_demo/hipblaslt_offline_tuning.ipynb) |
| **Lance × vLLM-Omni** unified multimodal Gradio demo | [lance_demo/README.md](lance_demo/README.md) | [lance_demo/run_gradio_demo.sh](lance_demo/run_gradio_demo.sh) |
