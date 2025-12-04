#!/bin/bash
# Usage: ./train_adm.sh <GPU_ID>
set -e

GPU_ID=$1
GM=adm
TRUNCATE_LAYER=24
EXPNAME=adm-lad
DATA_ROOT="/path/to/genimage"  # Set your data directory here
ARCH="CLIP:openai/clip-vit-large-patch14"

echo "=================================="
echo "GPU_ID: ${GPU_ID}"
echo "EXPNAME: ${EXPNAME}"
echo "TRUNCATE_LAYER: ${TRUNCATE_LAYER}"
echo "GM: ${GM}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "=================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train.py \
  --batch_size 128 --fix_backbone --niter 100 --arch ${ARCH} \
  --data_mode custom --data_root ${DATA_ROOT} \
  --gm ${GM} --expname ${EXPNAME} --truncate_layer ${TRUNCATE_LAYER}
