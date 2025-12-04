#!/bin/bash
# Usage: ./train_progan.sh <GPU_ID> <TRUNCATE_LAYER>
set -e

GPU_ID=$1
TRUNCATE_LAYER=$2
GM=4class
EXPNAME=progan-4class-l${TRUNCATE_LAYER}
ARCH="CLIP:openai/clip-vit-large-patch14"

echo "=================================="
echo "GPU_ID: ${GPU_ID}"
echo "EXPNAME: ${EXPNAME}"
echo "TRUNCATE_LAYER: ${TRUNCATE_LAYER}"
echo "GM: ${GM}"
echo "=================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train.py \
  --batch_size 128 --fix_backbone --niter 100 --arch ${ARCH} \
  --data_mode progan_custom --gm ${GM} --expname ${EXPNAME} \
  --truncate_layer ${TRUNCATE_LAYER}
