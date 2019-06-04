#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."


export QE_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/QE_en2de2017
export BERT_PRE_TRAINED_MODEL_DIR_SRC=/home/user_data/houq/bert_model/bert-base-cased
export BERT_PRE_TRAINED_MODEL_DIR_MT=/home/user_data/houq/bert_model/bert-base-multilingual-cased

python -m test_rnn \
  --task_name QE \
  --input_file=$QE_DIR/ \
  --bert_model_src $BERT_PRE_TRAINED_MODEL_DIR_SRC/ \
  --bert_model_mt $BERT_PRE_TRAINED_MODEL_DIR_MT/ \
  --layers=-1 \
  --max_seq_length=50 \
  --test_batch_size 32 \
  --output_dir_src ./qe_src_en2de2017_train_src2mt_biLSTM_adam_finetuning_no_common_multilingual_output_3_bs_32_dropout/ \
  --output_dir_mt ./qe_mt_en2de2017_train_src2mt_biLSTM_adam_finetuning_no_common_multilingual_output_3_bs_32_dropout/ \
  --output_dir_qe_model ./qe_model_en2de2017_train_src2mt_biLSTM_adam_finetuning_no_common_multilingual_output_3_bs_32_dropout/qe_bert.best.1100\
  --saveto "./result_en2de2017_train_src2mt_biLSTM_adam_finetuning_no_common_multilingual_output_3_bs_32_dropout/QE_en2de2017_test2017_predict.hter" \