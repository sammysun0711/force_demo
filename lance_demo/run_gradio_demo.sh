#!/bin/bash
#  --ulysses-degree 4

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python vllm-omni/examples/offline_inference/lance/gradio_demo.py \
  --model /models/Lance/ 
