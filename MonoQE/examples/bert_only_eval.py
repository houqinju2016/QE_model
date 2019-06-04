# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import re
import logging
import argparse
import random
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import time
import copy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'


class Timer(object):
    def __init__(self):
        self.t0 = 0

    def tic(self):
        self.t0 = time.time()

    def toc(self, format='m:s', return_seconds=False):
        t1 = time.time()

        if return_seconds is True:
            return t1 - self.t0

        if format == 's':
            return '{0:d}'.format(t1 - self.t0)
        m, s = divmod(t1 - self.t0, 60)
        if format == 'm:s':
            return '%d:%02d' % (m, s)
        h, m = divmod(m, 60)
        return '%d:%02d:%02d' % (h, m, s)


class Collections(object):
    """Collections for logs during training.

    Usually we add loss and valid metrics to some collections after some steps.
    """
    _MY_COLLECTIONS_NAME = "my_collections"

    def __init__(self, kv_stores=None, name=None):

        self._kv_stores = kv_stores if kv_stores is not None else {}

        if name is None:
            name = Collections._MY_COLLECTIONS_NAME
        self._name = name

    def add_to_collection(self, key, value):
        """
        Add value to collection

        :type key: str
        :param key: Key of the collection

        :param value: The value which is appended to the collection
        """
        if key not in self._kv_stores:
            self._kv_stores[key] = [value]
        else:
            self._kv_stores[key].append(value)

    def get_collection(self, key, default=[]):
        """
        Get the collection given a key

        :type key: str
        :param key: Key of the collection
        """
        if key not in self._kv_stores:
            return default
        else:
            return self._kv_stores[key]

    def state_dict(self):

        return self._kv_stores

    def load_state_dict(self, state_dict):

        self._kv_stores = copy.deepcopy(state_dict)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# features.append(
#             InputFeatures(input_ids_src=input_ids_src,
#                           input_mask_src=input_mask_src,
#                           segment_ids_src=segment_ids_src,
#                           input_ids_mt=input_ids_mt,
#                           input_mask_mt=input_mask_mt,
#                           segment_ids_mt=segment_ids_mt,
#                           label_id=label_id))
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_src, input_mask_src, segment_ids_src, input_ids_mt, input_mask_mt, segment_ids_mt,
                 label_id):
        self.input_ids_src = input_ids_src
        self.input_mask_src = input_mask_src
        self.segment_ids_src = segment_ids_src
        self.input_ids_mt = input_ids_mt
        self.input_mask_mt = input_mask_mt
        self.segment_ids_mt = segment_ids_mt
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MyProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.src2mt.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.src2mt.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, "train.src2mt.tsv")), "test2017")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = float(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples_test(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer_src, tokenizer_mt):
    """Loads a data file into a list of `InputBatch`s."""

    # label_map = {label: i for i, label in enumerate(label_list)}
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer_src.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens_src = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_src = [0] * len(tokens_src)
        input_ids_src = tokenizer_src.convert_tokens_to_ids(tokens_src)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask_src = [1] * len(input_ids_src)

        # Zero-pad up to the sequence length.
        padding_src = [0] * (max_seq_length - len(input_ids_src))
        input_ids_src += padding_src
        input_mask_src += padding_src
        segment_ids_src += padding_src

        tokens_b = tokenizer_mt.tokenize(example.text_b)
        if len(tokens_b) > max_seq_length - 2:
            tokens_b = tokens_b[:(max_seq_length - 2)]

        tokens_mt = ["[CLS]"] + tokens_b + ["[SEP]"]
        segment_ids_mt = [0] * len(tokens_mt)
        input_ids_mt = tokenizer_mt.convert_tokens_to_ids(tokens_mt)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask_mt = [1] * len(input_ids_mt)

        # Zero-pad up to the sequence length.
        padding_mt = [0] * (max_seq_length - len(input_ids_mt))
        input_ids_mt += padding_mt
        input_mask_mt += padding_mt
        segment_ids_mt += padding_mt

        # label_id = label_map[example.label]
        label_id = example.label

        features.append(
            InputFeatures(input_ids_src=input_ids_src,
                          input_mask_src=input_mask_src,
                          segment_ids_src=segment_ids_src,
                          input_ids_mt=input_ids_mt,
                          input_mask_mt=input_mask_mt,
                          segment_ids_mt=segment_ids_mt,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


#
# def read_examples(input_file):
#     """Read a list of `InputExample`s from an input file."""
#     examples = []
#     unique_id = 0
#     with open(input_file, "r", encoding='utf-8') as reader:
#         while True:
#             line = reader.readline()
#             if not line:
#                 break
#             line = line.strip()
#             text_a = None
#             text_b = None
#             m = re.match(r"^(.*) \|\|\| (.*)$", line)
#             if m is None:
#                 text_a = line
#             else:
#                 text_a = m.group(1)
#                 text_b = m.group(2)
#             examples.append(
#                 InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
#             unique_id += 1
#     return examples


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--saveto", type=str,
                        help="""Result prefix.""")

    parser.add_argument("--bert_model_src", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model_mt", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    parser.add_argument("--output_dir_src",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--output_dir_mt",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir_fc",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--log_path', type=str, default="./log",
                        help="The path for saving tensorboard logs. Default is ./log")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    processors = {
        "qe": MyProcessor
    }
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    model_collections = Collections()
    tokenizer_src = BertTokenizer.from_pretrained(args.bert_model_src, do_lower_case=args.do_lower_case)
    tokenizer_mt = BertTokenizer.from_pretrained(args.bert_model_mt, do_lower_case=args.do_lower_case)

    # Load a trained model and config that you have fine-tuned
    output_config_file_src = os.path.join(args.output_dir_src, CONFIG_NAME)
    config_src = BertConfig(output_config_file_src)
    model_src = BertModel(config_src)

    output_model_file_src = os.path.join(args.output_dir_src, WEIGHTS_NAME)
    model_state_dict_src = torch.load(output_model_file_src)
    model_src.load_state_dict(model_state_dict_src)

    # Load a trained model and config that you have fine-tuned
    output_config_file_mt = os.path.join(args.output_dir_mt, CONFIG_NAME)
    config_mt = BertConfig(output_config_file_mt)
    model_mt = BertModel(config_mt)

    output_model_file_mt = os.path.join(args.output_dir_mt, WEIGHTS_NAME)
    model_state_dict_mt = torch.load(output_model_file_mt)
    model_mt.load_state_dict(model_state_dict_mt)

    # Prepare model
    # model_src = BertModel.from_pretrained(args.bert_model_src)
    # model_mt = BertModel.from_pretrained(args.bert_model_mt)
    # load config
    # src.config==mt.config
    config_file = os.path.join(args.bert_model_src, CONFIG_NAME)
    config_src = BertConfig.from_json_file(config_file)

    model_src.to(device)
    model_mt.to(device)
    model_src.eval()
    model_mt.eval()

    # fnn
    full_connect = torch.nn.Linear(2 * config_src.hidden_size, 1)
    model_state_dict_fc = torch.load(args.output_dir_fc)
    full_connect.load_state_dict(model_state_dict_fc)
    full_connect.to(device)
    full_connect.eval()
    # sigmoid
    sigmoid = torch.nn.Sigmoid()
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer_src, tokenizer_mt)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids_src = torch.tensor([f.input_ids_src for f in eval_features], dtype=torch.long)
        all_input_mask_src = torch.tensor([f.input_mask_src for f in eval_features], dtype=torch.long)
        all_segment_ids_src = torch.tensor([f.segment_ids_src for f in eval_features], dtype=torch.long)

        all_input_ids_mt = torch.tensor([f.input_ids_mt for f in eval_features], dtype=torch.long)
        all_input_mask_mt = torch.tensor([f.input_mask_mt for f in eval_features], dtype=torch.long)
        all_segment_ids_mt = torch.tensor([f.segment_ids_mt for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids_src, all_input_mask_src, all_segment_ids_src,
                                  all_input_ids_mt, all_input_mask_mt, all_segment_ids_mt)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model_src.eval()
        model_mt.eval()
        full_connect.eval()

        predict_hter = []
        for bacth_eval in eval_dataloader:
            bacth_eval = tuple(t.to(device) for t in bacth_eval)
            input_ids_src, input_mask_src, segment_ids_src, input_ids_mt, \
            input_mask_mt, segment_ids_mt = bacth_eval
            with torch.no_grad():
                _, pooled_output_src = model_src(input_ids_src, segment_ids_src, input_mask_src,
                                                 output_all_encoded_layers=False)
                _, pooled_output_mt = model_mt(input_ids_mt, segment_ids_mt, input_mask_mt,
                                               output_all_encoded_layers=False)
                # pooled_output [batch_size,2*hidden_size]
                pooled_output = torch.cat((pooled_output_src, pooled_output_mt), 1)
                logits = sigmoid(full_connect(pooled_output.detach()))
            logits = logits.detach().squeeze(1).cpu().numpy().tolist()
            for val in logits:
                predict_hter.append(float(val))

        def auto_mkdir(path):
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        auto_mkdir('/'.join(args.saveto.split('/')[0:2]))
        with open(args.saveto, 'w') as f:
            for val in predict_hter:
                f.write('%.6f\n' % val)


if __name__ == "__main__":
    main()
