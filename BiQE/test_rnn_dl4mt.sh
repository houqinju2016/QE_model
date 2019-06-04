#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
N=$1
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_rnn \
    --model_name "qe"\
    --source_path "/home/user_data/houq/transformer_share/QE_en2de2017_formal/test2016.tok.tc.bpe.90000.src" \
    --target_path "/home/user_data/houq/transformer_share/QE_en2de2017_formal/test2016.tok.tc.bpe.90000.mt" \
    --transformer_model_path "./save_qe_en2de2017_adam_0.0001_1024_finetuning_patience10_batch32_emb_dl4mt17_formal_eval/dl4mt.best." \
    --qe_model_path "./save_qe_en2de2017_adam_0.0001_1024_finetuning_patience10_batch32_emb_dl4mt17_formal_eval/qe.best." \
    --config_path "./configs/rnn_en2de_wmt17_dl4mt.yaml" \
    --batch_size 32 \
    --saveto "./result_qe_en2de2017_adam_0.0001_1024_finetuning_patience10_batch32_emb_dl4mt17_formal_eval/QE_en2de2017_test2016_predict.hter" \
    --use_gpu