#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."


export QE_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/dl4mt_en2de2017
export BERT_PRE_TRAINED_MODEL_DIR=/home/user_data/houq/bert_model/bert-base-multilingual-cased

python -m examples.run_classifier_train \
  --task_name QE \
  --do_train \
  --do_eval \
  --data_dir $QE_DIR/ \
  --bert_model $BERT_PRE_TRAINED_MODEL_DIR \
  --max_seq_length 100 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./qe_en2de2017_dl4mt_nmt2mt_finetuning_common_multilingual_output_3_bs_32/ \
  --log_path "./log_en2de2017_dl4mt_nmt2mt_finetuning_common_multilingual_3_bs_32" \