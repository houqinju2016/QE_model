#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
    --model_name "transformer" \
    --reload \
    --config_path "configs/transformer_wmt17_en2de.yaml" \
    --log_path "./log_en2de_wmt17_bpe" \
    --saveto "./save_en2de_wmt17_bpe/" \
    --use_gpu \
