#!/bin/bash
# Usage: ./train_adm.sh <gpu_id>
set -e

GM=adm
gpu_id=$1
truncate_layer=24
expname=adm-lad

echo "=================================="
echo "gpu_id: ${gpu_id}"
echo "expname: ${expname}"
echo "truncate_layer: ${truncate_layer}"
echo "GM: ${GM}"
echo "=================================="

# ARCH=CLIP:ViT-L/14  # Legacy ipmlementation.
# ARCH="CLIP:openai/clip-vit-large-patch14"
ARCH="DINOv3:facebook/dinov3-vit7b16-pretrain-lvd1689m"

CUDA_VISIBLE_DEVICES=$gpu_id python scripts/train.py \
  --batch_size 128 --fix_backbone --niter 100 --arch ${ARCH} \
  --data_mode custom \
  --gm ${GM} --expname ${expname} --truncate_layer ${truncate_layer}
