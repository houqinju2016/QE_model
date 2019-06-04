#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train_rnn_bt \
    --transformer_model_name "transformer" \
    --transformer_model_bt_name "transformer_bt" \
    --qe_model_name "qe_attention" \
    --config_path "configs/rnn_attention_en2de_wmt17.yaml" \
    --log_path "./log_qe_attention_concat_en2de2017_adam_0.0001_1024_finetuning_patience10_batch28_fv_transformer17_formal_1" \
    --model_path "/home/user_data/houq/transformer_share/save/transformer.best.73000" \
    --model_bt_path "/home/user_data/houq/transformer_share/save/transformer.best.72000" \
    --saveto_transformer_model "./save_qe_attention_concat_en2de2017_adam_0.0001_1024_finetuning_patience10_batch28_fv_transformer17_formal_1/" \
    --saveto_transformer_model_bt "./save_qe_attention_concat_en2de2017_adam_0.0001_1024_finetuning_patience10_batch28_fv_transformer17_formal_1/" \
    --saveto_qe_model "./save_qe_attention_concat_en2de2017_adam_0.0001_1024_finetuning_patience10_batch28_fv_transformer17_formal_1/" \
    --use_gpu