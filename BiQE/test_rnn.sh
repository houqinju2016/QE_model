#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
N=$1
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.test_rnn \
    --model_name "qe"\
    --source_path "/home/user_data/houq/transformer_share/QE_en2de2019/train.tok.tc.bpe.90000.src" \
    --target_path "/home/user_data/houq/transformer_share/QE_en2de2019/train.tok.tc.bpe.90000.mt" \
    --transformer_model_path "./save_qe_en2de2019_adam_0.0001_1024_finetuning_patience10_batch16*2_fv_transformer17_formal_eval/transformer.best.1300" \
    --qe_model_path "./save_qe_en2de2019_adam_0.0001_1024_finetuning_patience10_batch16*2_fv_transformer17_formal_eval/qe.best.1300" \
    --config_path "./configs/rnn_en2de_wmt17.yaml" \
    --batch_size 16 \
    --saveto "./result_qe_en2de2019_adam_0.0001_1024_finetuning_patience10_batch16*2_fv_transformer17_formal_eval/QE_en2de2019_train_predict.hter" \
    --use_gpu