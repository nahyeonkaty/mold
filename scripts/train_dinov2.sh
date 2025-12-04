#!/bin/bash
# Usage: ./train_dinov2.sh <gpu_id>
set -e

gm=adm
gpu_id=$1
expname=adm-dinov2-base-learnable
truncate_layer=24

echo "=================================="
echo "gpu_id: ${gpu_id}"
echo "expname: ${expname}"
echo "truncate_layer: ${truncate_layer}"
echo "gm: ${gm}"
echo "=================================="

CUDA_VISIBLE_DEVICES=$gpu_id python scripts/train.py \
--batch_size 128 --fix_backbone --niter 100 --arch DINOv2:BASE \
--data_mode custom --gm ${gm} --expname ${expname} --truncate_layer ${truncate_layer}
