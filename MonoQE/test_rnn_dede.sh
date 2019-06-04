#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."


export QE_DIR=/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/dl4mt_en2de2017
export BERT_PRE_TRAINED_MODEL_DIR_SRC=/home/user_data/houq/bert_model/bert-base-multilingual-cased
export BERT_PRE_TRAINED_MODEL_DIR_MT=/home/user_data/houq/bert_model/bert-base-multilingual-cased

python -m test_rnn \
  --input_file=$QE_DIR/ \
  --bert_model_src $BERT_PRE_TRAINED_MODEL_DIR_SRC/ \
  --bert_model_mt $BERT_PRE_TRAINED_MODEL_DIR_MT/ \
  --qe_model_path "/home/houq/pytorch_bert/pytorch-pretrained-BERT-master/qe_exact_feature_en2de2017_dede_mt2nmt_output/qe_bert.best.16200" \
  --layers=-1 \
  --max_seq_length=50 \
  --test_batch_size 24 \
  --saveto "./result_exact_feature_en2de2017_dede_mt2nmt_output/QE_en2de2017_train_predict.hter" \