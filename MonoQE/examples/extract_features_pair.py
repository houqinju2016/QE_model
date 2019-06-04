# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import argparse
import collections
import logging
import json
import re
import numpy as np
import os
import sys
import random
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import copy
from tempfile import mkstemp
from subprocess import call
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME

from src.models.qe import QE, QE_PAIR
from src.optim import Optimizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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
            self._read_tsv(os.path.join(data_dir, "train.src2mt.ori.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.src2mt.ori.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_test(
            self._read_tsv(os.path.join(data_dir, "test.src2mt.ori.tsv")), "test2017")

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


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        label_id = example.label
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %f (id = %f)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
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


def load_label_data(data_path):
    fy = open(data_path, 'r')
    array = fy.readlines()
    listy = []

    for line in array:
        line = np.float32(line.strip())
        listy.append(line)
    return np.array(listy)


def shuffle(files):
    tf_os, tpath = mkstemp()
    tf = open(tpath, 'w')

    fds = [open(ff) for ff in files]

    for l in fds[0]:
        lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
        tf.write("|||".join(lines) + "\n")

    [ff.close() for ff in fds]
    tf.close()

    tf = open(tpath, 'r')
    lines = tf.readlines()
    random.shuffle(lines)

    fds = [open(ff + '.shuf', 'w') for ff in files]

    for l in lines:
        s = l.strip().split('|||')
        for ii, fd in enumerate(fds):
            fd.write(s[ii] + "\n")

    [ff.close() for ff in fds]

    os.remove(tpath)


def compute_forward_qe_bt(model, critic, feature, data, label, eval=False):
    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            # predict label
            outputs = model(feature, data)
            loss = critic(outputs.view(-1), label.view(-1))
            # print("outputs.requires_grad")
            # print(outputs.requires_grad)
            # print("loss.requires_grad")
            # print(loss.requires_grad)

        # torch.autograd.backward(loss)
        loss.backward()
        return loss.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            # predict label
            outputs = model(feature, data)
            loss = critic(outputs.view(-1), label.view(-1))
        return loss.item()


def train_rnn(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))
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
    layer_indexes = [int(x) for x in args.layers.split(",")]

    # 0. load bert
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)
    # model.eval()
    # 1. Build Bi-LSTM Model & Criterion
    qe_model = QE_PAIR(feature_size=768, hidden_size=512, dropout_rate=0.1)
    qe_model.to(device)
    # ---------------------------------------------------------------------
    # fine-tuning fine-tuning model
    # Load a trained model and config that you have fine-tuned
    # output_config_file = os.path.join(args.bert_model, CONFIG_NAME)
    # config = BertConfig(output_config_file)
    # model = BertModel(config)
    #
    # output_model_file = os.path.join(args.bert_model, WEIGHTS_NAME)
    # model_state_dict = torch.load(output_model_file)
    # model.load_state_dict(model_state_dict)
    # model.to(device)
    #
    # # 1. load Bi-LSTM Model
    # qe_model = QE_PAIR(feature_size=768, hidden_size=512, dropout_rate=0.1)
    # qe_model_state_dict = torch.load(args.qe_model)
    # qe_model.load_state_dict(qe_model_state_dict)
    # qe_model.to(device)
    # --------------------------------------------------------------------
    critic = torch.nn.MSELoss()
    critic.to(device)
    # 2. Build optimizer
    train_examples = processor.get_train_examples(args.input_file)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    param_optimizer = list(model.named_parameters()) + list(qe_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    # optimizer=torch.optim.Adam([{'params':model.parameters()},
    #                             {'params':qe_model.parameters(),'lr':args.learning_rate*10}],
    #                             lr=args.learning_rate)

    # optimizer = "adam"
    # lrate = 2e-5
    # grad_clip = 1.0
    # optimizer = Optimizer(name=optimizer, model=qe_model, lr=lrate,
    #                   grad_clip=grad_clip, optim_args=None)

    # 3. prepare training data
    train_features = convert_examples_to_features(
        examples=train_examples, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids_train = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask_train = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids_train = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids_train = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

    train = TensorDataset(all_input_ids_train, all_input_mask_train, all_segment_ids_train, all_label_ids_train)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train)
    else:
        train_sampler = DistributedSampler(train)

    train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)
    summary_writer = SummaryWriter(log_dir=args.log_path)
    is_early_stop = False
    disp_freq = 100
    loss_valid_freq = 100
    early_stop_patience = 10
    bad_count = 0
    # 3. begin training
    eidx = 0
    uidx = 0
    while True:
        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(train_dataloader),
                                     unit="sents")
        for batch_train in train_dataloader:
            model.train()
            qe_model.train()
            uidx += 1
            batch_train = tuple(t.to(device) for t in batch_train)
            input_ids_train, input_mask_train, segment_ids_train, label_ids_train = batch_train
            n_samples_t = len(input_ids_train)
            training_progress_bar.update(n_samples_t)

            optimizer.zero_grad()
            # with torch.no_grad():
            #     all_encoder_layers_train, _ = model(input_ids_train, token_type_ids=segment_ids_train,
            #                                         attention_mask=input_mask_train)
            #     layer_output_train = all_encoder_layers_train[-1]
            #     layer_output_train_detach=layer_output_train.detach()
            with torch.enable_grad():
                all_encoder_layers_train, _ = model(input_ids_train, token_type_ids=segment_ids_train,
                                                    attention_mask=input_mask_train)
                layer_output_train = all_encoder_layers_train[-1]
                loss = compute_forward_qe_bt(model=qe_model, critic=critic,
                                             feature=layer_output_train,
                                             data=input_ids_train,
                                             label=label_ids_train,
                                             eval=False)
            optimizer.step()
            if (uidx % disp_freq == 0):
                lrate = args.learning_rate * warmup_linear(
                    uidx / t_total, args.warmup_proportion)
                result = {'train_loss': loss, 'lrate': lrate}
                # result = {'train_loss': loss}
                logger.info("***** train results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
            # calculate dev loss
            if (uidx % loss_valid_freq == 0):
                eval_examples = processor.get_dev_examples(args.input_file)
                eval_features = convert_examples_to_features(
                    examples=eval_examples, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                all_input_ids_eval = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask_eval = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids_eval = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids_eval = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

                eval = TensorDataset(all_input_ids_eval, all_input_mask_eval, all_segment_ids_eval,
                                     all_label_ids_eval)
                eval_sampler = SequentialSampler(eval)
                eval_dataloader = DataLoader(eval, sampler=eval_sampler, batch_size=args.eval_batch_size)
                sum_loss = 0.0
                nb_eval_steps = 0

                model.eval()
                qe_model.eval()

                for batch_eval in eval_dataloader:
                    batch_eval = tuple(t.to(device) for t in batch_eval)
                    input_ids_eval, input_mask_eval, input_segment_ids_eval, label_ids_eval = batch_eval
                    with torch.no_grad():
                        all_encoder_layers_eval, _ = model(input_ids_eval, token_type_ids=input_segment_ids_eval,
                                                           attention_mask=input_mask_eval)
                        layer_output_eval = all_encoder_layers_eval[-1]
                        eval_loss_batch = compute_forward_qe_bt(model=qe_model, critic=critic,
                                                                feature=layer_output_eval,
                                                                data=input_ids_eval,
                                                                label=label_ids_eval,
                                                                eval=True)
                    if np.isnan(eval_loss_batch):
                        logging.info("NaN detected!")
                    sum_loss += float(eval_loss_batch)
                    nb_eval_steps += 1

                eval_loss = sum_loss / nb_eval_steps
                model_collections.add_to_collection("history_losses", eval_loss)
                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()
                summary_writer.add_scalar("loss", eval_loss, global_step=uidx)
                summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)
                lrate = args.learning_rate * warmup_linear(
                    uidx / t_total, args.warmup_proportion)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                best_eval_loss = min_history_loss
                # If model get new best valid loss
                # save model & early stop
                if eval_loss <= best_eval_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # Save a trained model and the associated configuration
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(args.output_dir_model, WEIGHTS_NAME)
                        torch.save(model_to_save.state_dict(), output_model_file)

                        output_config_file = os.path.join(args.output_dir_model, CONFIG_NAME)
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())

                        # Save a qe_model
                        output_qe_model_file = os.path.join(args.output_dir_qe_model, "qe_bert.best." + str(uidx))
                        torch.save(qe_model.state_dict(), output_qe_model_file)
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= early_stop_patience and eidx > 0:
                        is_early_stop = True
                        logger.info("Early Stop!")
                summary_writer.add_scalar("bad_count", bad_count, uidx)

                logger.info("{0} Loss: {1:.14f}   patience: {2}".format(
                    uidx, eval_loss, bad_count))
                print("{0} Loss: {1:.14f}   patience: {2}".format(
                    uidx, eval_loss, bad_count))
            if is_early_stop == True:
                break
        training_progress_bar.close()
        eidx += 1
        if eidx > args.num_train_epochs:
            break


def test_rnn(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))
    layer_indexes = [int(x) for x in args.layers.split(",")]

    processors = {
        "qe": MyProcessor
    }
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    # 0. load bert
    # Load a trained model and config that you have fine-tuned
    output_config_file = os.path.join(args.output_dir_model, CONFIG_NAME)
    config = BertConfig(output_config_file)
    model = BertModel(config)

    output_model_file = os.path.join(args.output_dir_model, WEIGHTS_NAME)
    model_state_dict = torch.load(output_model_file)
    model.load_state_dict(model_state_dict)

    # output_model_file = os.path.join(args.output_dir_model, WEIGHTS_NAME)
    # model_state_dict = torch.load(output_model_file)
    # model = BertModel.from_pretrained(args.bert_model, state_dict=model_state_dict)
    # model = BertModel.from_pretrained(args.bert_model)
    model.to(device)
    model.eval()

    # 1. load Bi-LSTM Model
    qe_model = QE_PAIR(feature_size=768, hidden_size=512, dropout_rate=0.0)
    qe_model_state_dict = torch.load(args.output_dir_qe_model)
    qe_model.load_state_dict(qe_model_state_dict)

    qe_model.to(device)
    qe_model.eval()
    # 2. prepare test data
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    test_examples = processor.get_test_examples(args.input_file)
    test_features = convert_examples_to_features(
        examples=test_examples, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
    all_input_ids_test = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask_test = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids_test = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

    test = TensorDataset(all_input_ids_test, all_input_mask_test, all_segment_ids_test)
    # no shuffle
    test_sampler = SequentialSampler(test)
    test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.test_batch_size)

    result = []
    for batch_test in test_dataloader:
        batch_test = tuple(t.to(device) for t in batch_test)
        input_ids_test, input_mask_test, segment_ids_test = batch_test
        with torch.no_grad():
            all_encoder_layers_test, _ = model(input_ids_test, token_type_ids=segment_ids_test,
                                               attention_mask=input_mask_test)
            layer_output_test = all_encoder_layers_test[-1]
            # predict label
            outputs = qe_model(layer_output_test, input_ids_test)
        outputs.squeeze(1).cpu().numpy().tolist()
        for val in outputs:
            result.append(float(val))
    with open(args.saveto, 'w') as f:
        for val in result:
            f.write('%.6f\n' % val)
