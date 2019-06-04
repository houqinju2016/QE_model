#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=2

N=$1

#export THEANO_FLAGS=device=cuda3,floatX=float32,mode=FAST_RUN
export MODEL_NAME="transformer"

python ./teacher_translate.py \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/weihr/NMT_DATA_PY3/WMT14-DE-EN/newstest2013.tok.bpe.32000.en" \
    --target_path "/home/user_data/weihr/NMT_DATA_PY3/WMT14-DE-EN/newstest2013.tok.bpe.32000.en" \
    --model_path "./$MODEL_NAME.best.45000" \
    --config_path "./configs/transformer_wmt14_en2de.yaml" \
    --batch_size 20 \
    --saveto "./result/$MODEL_NAME.newstest2013.txt" \
    --source_bpe_codes "" \