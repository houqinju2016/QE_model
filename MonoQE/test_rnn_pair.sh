#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."


export QE_DIR=/home/user_data/houq/transformer_share/QE_en2de2019/

export BERT_PRE_TRAINED_MODEL_DIR=/home/user_data/houq/bert_model/bert-base-multilingual-cased

python -m test_rnn_pair \
  --task_name QE \
  --input_file=$QE_DIR/ \
  --bert_model $BERT_PRE_TRAINED_MODEL_DIR/ \
  --output_dir_model ./model_en2de2019_src2mt_ori_biLSTM_adam_finetuning_common_multilingual_3_bs_16_dropout/ \
  --output_dir_qe_model ./qe_model_en2de2019_src2mt_ori_biLSTM_adam_finetuning_common_multilingual_3_bs_16_dropout/qe_bert.best.900 \
  --layers=-1 \
  --max_seq_length=100 \
  --test_batch_size 16 \
  --saveto "./result_en2de2019_src2mt_ori_biLSTM_adam_finetuning_common_multilingual_3_bs_16_dropout/QE_en2de2019_test_predict.hter" \