#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
N=$1

#export THEANO_FLAGS=device=cuda3,floatX=float32,mode=FAST_RUN
#export MODEL_NAME="transformer"

python -m src.bin.test_rnn_bt \
    --model_name "qe_bt"\
    --source_path "/home/user_data/houq/transformer_share/QE_en2de2017_formal/dev.tok.tc.bpe.90000.src" \
    --target_path "/home/user_data/houq/transformer_share/QE_en2de2017_formal/dev.tok.tc.bpe.90000.mt" \
    --transformer_model_path "/home/user_data/houq/transformer_share/save/dl4mt.best.33000" \
    --transformer_model_bt_path "/home/user_data/houq/transformer_share/save/dl4mt.best.35500" \
    --qe_model_path "./save_qe_bt_en2de2017_adam_0.001_1024_patience10_batch32_emb_dl4mt17_formal/qe_bt.best." \
    --config_path "./configs/rnn_bt_en2de_wmt17_dl4mt.yaml" \
    --batch_size 20 \
    --saveto "./result_qe_bt_en2de2017_adam_0.001_1024_patience10_batch32_emb_dl4mt17_formal/QE_en2de2017_dev_predict.hter" \
    --use_gpu