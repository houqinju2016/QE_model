#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."


export QE_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/transformer_en2de2017
export BERT_PRE_TRAINED_MODEL_DIR_SRC=/home/user_data/houq/bert_model/bert-base-multilingual-cased
export BERT_PRE_TRAINED_MODEL_DIR_MT=/home/user_data/houq/bert_model/bert-base-multilingual-cased

python -m train_rnn \
  --input_file=$QE_DIR/ \
  --bert_model_src $BERT_PRE_TRAINED_MODEL_DIR_SRC/ \
  --bert_model_mt $BERT_PRE_TRAINED_MODEL_DIR_MT/ \
  --layers=-1 \
  --max_seq_length=50 \
  --train_batch_size=24 \
  --eval_batch_size=24 \
  --num_train_epochs 50.0 \
  --output_dir ./qe_exact_feature_en2de2017_dede_transformer_mt2nmt_output/ \
  --log_path ./log_exact_feature_en2de2017_dede_transformer_mt2nmt/ \