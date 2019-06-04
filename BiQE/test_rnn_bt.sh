#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
N=$1

#export THEANO_FLAGS=device=cuda3,floatX=float32,mode=FAST_RUN
#export MODEL_NAME="transformer"

python -m src.bin.test_rnn_bt \
    --model_name "qe_bt"\
    --source_path "/home/user_data/houq/transformer_share/QE_en2de2018_SMT_formal/dev.tok.tc.bpe.90000.src" \
    --target_path "/home/user_data/houq/transformer_share/QE_en2de2018_SMT_formal/dev.tok.tc.bpe.90000.src" \
    --transformer_model_path "./save_qe_bt_en2de2018_SMT_adam_0.0001_1024_finetuning_patience10_batch16_hidden512_fv_transformer17_eval/transformer.best.1900" \
    --transformer_model_bt_path "./save_qe_bt_en2de2018_SMT_adam_0.0001_1024_finetuning_patience10_batch16_hidden512_fv_transformer17_eval/transformer_bt.best.1900" \
    --qe_model_path "./save_qe_bt_en2de2018_SMT_adam_0.0001_1024_finetuning_patience10_batch16_hidden512_fv_transformer17_eval/qe_bt.best.1900" \
    --config_path "./configs/rnn_bt_en2de_wmt17.yaml" \
    --batch_size 20 \
    --saveto "./result_qe_bt_en2de2018_SMT_adam_0.0001_1024_finetuning_patience10_batch16_hidden512_fv_transformer17_eval/QE_en2de2018_SMT_dev_predict.hter" \
    --use_gpu