#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
N=$1

#export THEANO_FLAGS=device=cuda3,floatX=float32,mode=FAST_RUN
#export MODEL_NAME="transformer"

python -m src.bin.test_rnn_bt \
    --model_name "qe_bt"\
    --source_path "/home/user_data/houq/transformer_share/QE_en2de2017_formal/dev.tok.tc.bpe.90000.src" \
    --target_path "/home/user_data/houq/transformer_share/QE_en2de2017_formal/dev.tok.tc.bpe.90000.mt" \
    --transformer_model_path "./save_qe_attention_concat_en2de2017_adam_0.0001_1024_finetuning_patience10_batch28_fv_transformer17_formal_1/transformer.best." \
    --transformer_model_bt_path "./save_qe_attention_concat_en2de2017_adam_0.0001_1024_finetuning_patience10_batch28_fv_transformer17_formal_1/transformer_bt.best." \
    --qe_model_path "./save_qe_attention_concat_en2de2017_adam_0.0001_1024_finetuning_patience10_batch28_fv_transformer17_formal_1/qe_attention.best." \
    --config_path "./configs/rnn_attention_en2de_wmt17.yaml" \
    --batch_size 20 \
    --saveto "./result_qe_attention_concat_en2de2017_adam_0.0001_1024_finetuning_patience10_batch28_fv_transformer17_formal_1/QE_en2de2017_dev_predict.hter" \
    --use_gpu