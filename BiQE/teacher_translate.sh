#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
N=$1
export MODEL_NAME="transformer"

python -m src.bin.teacher_translate \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/houq/wmt17_en_ru/dev.tok.tc.bpe.90000.ru" \
    --target_path "/home/user_data/houq/wmt17_en_ru/dev.tok.tc.bpe.90000.en" \
    --model_path "/home/houq/transformer_share/NJUNMT-pytorch-master/save_ru2en_wmt17_bpe/$MODEL_NAME.best.final" \
    --config_path "./configs/transformer_wmt17_ru2en.yaml" \
    --batch_size 20 \
    --saveto "./result_ru2en_wmt17_transformer_force_decode/$MODEL_NAME.dev.txt.en.58k" \
    --use_gpu