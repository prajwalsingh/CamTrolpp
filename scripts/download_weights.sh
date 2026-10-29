#!/bin/bash
# Download the two model checkpoints into checkpoints/ (about 14 GB in total).
# Stable Video Diffusion and SD-1.5 inpainting come with their own licenses,
# please read them on the model pages before use.
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints
huggingface-cli download stabilityai/stable-video-diffusion-img2vid svd.safetensors --local-dir checkpoints
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-inpainting sd-v1-5-inpainting.ckpt --local-dir checkpoints
ls -lh checkpoints
