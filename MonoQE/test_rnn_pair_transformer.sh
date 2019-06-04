#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."


export QE_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/bert_en2de2018_SMT

export BERT_PRE_TRAINED_MODEL_DIR=/home/user_data/houq/bert_model/bert-base-multilingual-cased

python -m test_rnn_pair \
  --task_name QE \
  --input_file=$QE_DIR/ \
  --bert_model $BERT_PRE_TRAINED_MODEL_DIR/ \
  --output_dir_model ./model_en2de2018_SMT_src2mt_biLSTM_adam_finetuning_common_multilingual_3_bs_32_dropout/ \
  --output_dir_qe_model ./qe_model_en2de2018_SMT_src2mt_biLSTM_adam_finetuning_common_multilingual_3_bs_32_dropout/qe_bert.best.1300 \
  --layers=-1 \
  --max_seq_length=100 \
  --test_batch_size 32 \
  --saveto "./result_en2de2018_SMT_src2mt_biLSTM_adam_finetuning_common_multilingual_3_bs_32_dropout/QE_en2de2018_SMT_test_predict.hter" \