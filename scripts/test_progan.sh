#!/bin/bash
# Usage: ./test_progan.sh <gpu_id> <gm>
set -e

gpu_id=$1
gm=$2
truncate_layer=24

ckpt=progan-final

echo "========================="
echo gpu_id: ${gpu_id}
echo ckpt: ${ckpt}
echo truncate_layer: ${truncate_layer}
echo gm: ${gm}
echo "========================="
CUDA_VISIBLE_DEVICES=${gpu_id} python3 scripts/validate.py --data_mode custom \
  --real_path /mnt/kaist1/pnh/datasets/progan/test/${gm}/real \
  --fake_path /mnt/kaist1/pnh/datasets/progan/test/${gm}/fake  \
  --ckpt=./checkpoints/${ckpt}/earlystop_best.pth \
  --arch=CLIP:ViT-L/14 --result_folder=clip_vitl14 --truncate_layer ${truncate_layer}
