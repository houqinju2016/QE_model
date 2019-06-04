#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_rnn_bt \
    --transformer_model_name "transformer" \
    --transformer_model_bt_name "transformer_bt" \
    --qe_model_name "qe_bt" \
    --config_path "configs/rnn_bt_en2de_wmt17.yaml" \
    --log_path "./log_qe_bt_en2de2018_SMT_adam_0.0001_1024_finetuning_patience10_batch16_hidden512_fv_transformer17_eval" \
    --model_path "/home/user_data/houq/transformer_share/save/transformer.en2de.73000" \
    --model_bt_path "/home/user_data/houq/transformer_share/save/transformer.de2en.72000" \
    --saveto_transformer_model "./save_qe_bt_en2de2018_SMT_adam_0.0001_1024_finetuning_patience10_batch16_hidden512_fv_transformer17_eval/" \
    --saveto_transformer_model_bt "./save_qe_bt_en2de2018_SMT_adam_0.0001_1024_finetuning_patience10_batch16_hidden512_fv_transformer17_eval/" \
    --saveto_qe_model "./save_qe_bt_en2de2018_SMT_adam_0.0001_1024_finetuning_patience10_batch16_hidden512_fv_transformer17_eval/" \
    --use_gpu