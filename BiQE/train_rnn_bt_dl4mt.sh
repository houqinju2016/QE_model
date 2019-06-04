#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_rnn_bt \
    --transformer_model_name "dl4mt" \
    --transformer_model_bt_name "dl4mt_bt" \
    --qe_model_name "qe_bt" \
    --config_path "./configs/rnn_bt_en2de_wmt17_dl4mt.yaml" \
    --log_path "./log_qe_bt_en2de2017_adam_0.0001_1024_finetuning_patience10_batch32_emb_dl4mt17_formal_eval" \
    --model_path "/home/user_data/houq/transformer_share/save/dl4mt.best.33000" \
    --model_bt_path "/home/user_data/houq/transformer_share/save/dl4mt.best.35500" \
    --saveto_transformer_model "./save_qe_bt_en2de2017_adam_0.001_1024_patience10_batch32_emb_dl4mt17_formal/" \
    --saveto_transformer_model_bt "./save_qe_bt_en2de2017_adam_0.001_1024_patience10_batch32_emb_dl4mt17_formal/" \
    --saveto_qe_model "./save_qe_bt_en2de2017_adam_0.001_1024_patience10_batch32_emb_dl4mt17_formal/" \
    --use_gpu