#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."
#--pretrain_path "./save_qe_en2de2017_large/qe.best.2600" \
#--qe_model_path "/home/user_data/houq/transformer_share/save/qe.best.700.patience10" \
#--saveto_transformer_model "./save_qe_en2de2017_adam_0.001_2048_finetuning_patience10_no_pretrain/" \
python -m src.bin.train_rnn \
    --transformer_model_name "dl4mt" \
    --qe_model_name "qe" \
    --config_path "./configs/rnn_en2de_wmt17_dl4mt.yaml" \
    --model_path "/home/user_data/houq/transformer_share/save/dl4mt.best.33000" \
    --saveto_transformer_model "./save_qe_en2de2017_adam_0.0001_1024_finetuning_patience10_batch32_emb_dl4mt17_formal_eval/" \
    --saveto_qe_model "./save_qe_en2de2017_adam_0.0001_1024_finetuning_patience10_batch32_emb_dl4mt17_formal_eval/" \
    --log_path "./log_qe_en2de2017_adam_0.0001_1024_finetuning_patience10_batch32_emb_dl4mt17_formal_eval / " \
    --use_gpu