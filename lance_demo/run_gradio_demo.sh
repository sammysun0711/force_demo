#!/bin/bash
#  --ulysses-degree 4

export CUDA_VISIBLE_DEVICES=5,6
python vllm-omni/examples/offline_inference/lance/gradio_demo.py \
  --model /models/Lance/ --port 32768
