#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
    --model_name "transformer" \
    --reload \
    --config_path "configs/transformer_wmt17_ru2en.yaml" \
    --log_path "./log_ru2en_wmt17_bpe" \
    --saveto "./save_ru2en_wmt17_bpe/" \
    --use_gpu \