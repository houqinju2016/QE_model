#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

export QE_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/bert_en2de2018_SMT
export BERT_PRE_TRAINED_MODEL_DIR=/home/user_data/houq/bert_model/bert-base-multilingual-cased
#--qe_model ./qe_model_en2de2017_large_src2mt_biLSTM_adam_finetuning_common_multilingual_3_bs_32_dropout/qe_bert.best.4000 \
#--output_dir_model ./model_en2de2018_SMT_nmt2mt_biLSTM_adam_finetuning_common_multilingual_5_bs_32/ \
python -m train_rnn_pair \
  --task_name QE \
  --input_file=$QE_DIR/ \
  --bert_model $BERT_PRE_TRAINED_MODEL_DIR \
  --layers=-1 \
  --max_seq_length=100 \
  --learning_rate 2e-5 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --num_train_epochs 3.0 \
  --output_dir_model ./model_en2de2018_SMT_src2mt_biLSTM_adam_finetuning_common_multilingual_3_bs_32_dropout/ \
  --output_dir_qe_model ./qe_model_en2de2018_SMT_src2mt_biLSTM_adam_finetuning_common_multilingual_3_bs_32_dropout/ \
  --log_path ./log_en2de2018_SMT_src2mt_biLSTM_adam_finetuning_common_multilingual_3_bs_32_dropout/ \