#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."


export QE_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/dl4mt_en2de2017
export BERT_PRE_TRAINED_MODEL_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/bert-base-multilingual-uncased

python -m examples.run_classifier \
  --task_name QE \
  --do_train \
  --do_eval \
  --data_dir $QE_DIR/ \
  --bert_model $BERT_PRE_TRAINED_MODEL_DIR/ \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 50.0 \
  --output_dir ./qe_de2en2017_output_logits_3_uncased/