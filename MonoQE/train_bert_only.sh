#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

echo "Using GPU $CUDA_VISIBLE_DEVICES..."
#--output_dir_src ./qe_src_en2de2017_src2mt_finetuning_no_common_multilingual_output/ \
#--output_dir_mt ./qe_mt_en2de2017_src2mt_finetuning_no_common_multilingual_output/ \

export QE_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/QE_en2de2017
export BERT_PRE_TRAINED_MODEL_DIR_SRC=/home/user_data/houq/bert_model/bert-base-cased
export BERT_PRE_TRAINED_MODEL_DIR_MT=/home/user_data/houq/bert_model/bert-base-multilingual-cased

python -m examples.bert_only_train \
  --task_name QE \
  --do_train \
  --do_eval \
  --data_dir $QE_DIR/ \
  --bert_model_src ./qe_src_en2de2017_large_src2mt_finetuning_no_common_multilingual_output_3_bs_32/ \
  --bert_model_mt ./qe_mt_en2de2017_large_src2mt_finetuning_no_common_multilingual_output_3_bs_32/ \
  --fc_model ./qe_fc_en2de2017_large_src2mt_finetuning_no_common_multilingual_output_3_bs_32/fnn.best.4200\
  --max_seq_length=50 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir_src ./qe_src_en2de2017_train_src2mt_finetuning_no_common_multilingual_output_3_bs_32/ \
  --output_dir_mt ./qe_mt_en2de2017_train_src2mt_finetuning_no_common_multilingual_output_3_bs_32/ \
  --output_dir_fc ./qe_fc_en2de2017_train_src2mt_finetuning_no_common_multilingual_output_3_bs_32/\
  --log_path ./log_en2de2017_train_src2mt_finetuning_no_common_multilingual_3_bs_32 \