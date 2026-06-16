export HIP_VISIBLE_DEVICES=7
python vllm-omni/examples/offline_inference/lance/end2end.py \
 --model /models/Lance \
 --prompts "a corgi astronaut floating in space above the moon, cinematic, photorealistic" \
 --steps 30 \
 --output ./out
