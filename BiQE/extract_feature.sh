#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
N=$1

#export THEANO_FLAGS=device=cuda3,floatX=float32,mode=FAST_RUN
export MODEL_NAME="transformer"

python3 -m src.bin.extract_feature \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/houq/transformer/QE_en2de2017/train.tok.bpe.32000.src" \
    --target_path "/home/user_data/houq/transformer/QE_en2de2017/train.tok.bpe.32000.mt" \
    --model_path "/home/user_data/houq/transformer/save/$MODEL_NAME.best.final" \
    --config_path "./configs/transformer_wmt14_en2de.yaml" \
    --batch_size 20 \
    --saveto_emb "./result/$MODEL_NAME.train_en2de2017.emb.txt" \
    --saveto_hidden "./result/$MODEL_NAME.train_en2de2017.hidden.txt" \
    --saveto_logit "./result/$MODEL_NAME.train_en2de2017.logit.txt" \
    --use_gpu