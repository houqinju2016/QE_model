#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
    --model_name "dl4mt" \
    --reload \
    --config_path "configs/dl4mt_wmt17_de2en.yaml" \
    --log_path "./log_de2en_wmt17_dl4mt" \
    --saveto "./save_de2en_wmt17_dl4mt/" \
    --use_gpu \
