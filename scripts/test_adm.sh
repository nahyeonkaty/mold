#!/bin/bash
# Usage: ./test_adm.sh <GPU_ID> <GM> <SKIP_IDX>
set -e

GPU_ID=$1
GM=$2
SKIP_IDX=$3

TRUNCATE_LAYER=24
DATA_ROOT="/path/to/genimage"  # Set your data directory here

echo "========================="
echo "GPU_ID: ${GPU_ID}"
echo "TRUNCATE_LAYER: ${TRUNCATE_LAYER}"
echo "GM: ${GM}"
echo "SKIP_IDX: ${SKIP_IDX}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "========================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 scripts/validate.py --data_mode custom \
  --real_path ${DATA_ROOT}/${GM}/val/real \
  --fake_path ${DATA_ROOT}/${GM}/val/fake \
  --ckpt=checkpoints/adm-lad/earlystop_best.pth \
  --arch=CLIP:openai/clip-vit-large-patch14 \
  --result_folder=results --truncate_layer ${TRUNCATE_LAYER} --skip_idx ${SKIP_IDX}
