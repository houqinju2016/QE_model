import os
import random
import time

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, QEDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.decoding import beam_search, ensemble_beam_search
from src.metric.bleu_scorer import SacreBLEUScorer
from src.models import build_model
from src.modules.criterions import NMTCriterion
from src.optim import Optimizer
from src.optim.lr_scheduler import ReduceOnPlateauScheduler, NoamScheduler
from src.utils.common_utils import *
from src.utils.logging import *
from src.utils.moving_average import MovingAverage
from src.utils.configs import default_configs, pretty_configs

BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def split_shard(*inputs, split_size=1):
    if split_size <= 1:
        yield inputs
    else:

        lengths = [len(s) for s in inputs[-1]]  #
        sorted_indices = np.argsort(lengths)

        # sorting inputs

        inputs = [
            [inp[ii] for ii in sorted_indices]
            for inp in inputs
            ]

        # split shards
        total_batch = sorted_indices.shape[0]  # total number of batches

        if split_size >= total_batch:
            yield inputs
        else:
            shard_size = total_batch // split_size

            _indices = list(range(total_batch))[::shard_size] + [total_batch]

            for beg, end in zip(_indices[:-1], _indices[1:]):
                yield (inp[beg:end] for inp in inputs)


def prepare_data(seqs_x, seqs_y=None, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.

    Returns:

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [BOS] + s + [EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=PAD,
                         cuda=cuda, batch_first=batch_first)

    return x, y


def compute_forward(model,
                    critic,
                    seqs_x,
                    seqs_y,
                    eval=False,
                    normalization=1.0,
                    norm_by_words=False):
    """
    :type model: nn.Module

    :type critic: NMTCriterion
    """
    # y_inp = seqs_y[:, :-1].contiguous()
    # y_label = seqs_y[:, 1:].contiguous()

    # ------------------houq--------------------------------
    y_inp = seqs_y[:, :].contiguous()
    y_label = seqs_y[:, :].contiguous()

    words_norm = y_label.ne(PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            # INFO('enter model computer forward...')
            log_probs = model(seqs_x, y_inp)
            # INFO('end model computer forward...')

            # INFO('enter critic KLDivLoss ...')
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            # INFO('end critic KLDivLoss ...')

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:

                # loss:(batch,)
                # div(batch_size)
                loss = loss.sum()
                # loss.sum() = loss on single sample
        torch.autograd.backward(loss)
        # get the value of a tensor(return number), use .item()
        return loss.item()
    else:
        # close dropout
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs = model(seqs_x, y_inp)

            # loss:(batch,)
            # div(1)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            # loss=loss.sum()
        return loss.item()


def compute_forward_qe(model,
                       critic,
                       feature,
                       y,
                       y_label,
                       eval=False,
                       ):
    """
    :type model: nn.Module

    :type critic: NMTCriterion
    """

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            # predict label
            outputs = model(feature, y)
            loss = critic(outputs, y_label)
        # torch.autograd.backward(loss)
        # loss.backward()
        return loss
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            # predict label
            outputs = model(feature, y)
            loss = critic(outputs, y_label)
        return loss.item()


def compute_forward_qe_bt(model,
                          critic,
                          feature,
                          feature_bt,
                          y,
                          x,
                          y_label,
                          eval=False,
                          ):
    """
    :type model: nn.Module

    :type critic: NMTCriterion
    """

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            # predict label
            outputs = model(feature, y, feature_bt, x)
            loss = critic(outputs, y_label)
        # torch.autograd.backward(loss)
        return loss
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            # predict label
            outputs = model(feature, y, feature_bt, x)
            loss = critic(outputs, y_label)
        return loss.item()


def loss_validation(model, critic, valid_iterator):
    """
    :type model: Transformer

    :type critic: NMTCriterion

    :type valid_iterator: DataIterator
    """

    n_sents = 0
    n_tokens = 0.0

    sum_loss = 0.0

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        _, seqs_x, seqs_y = batch

        n_sents += len(seqs_x)
        n_tokens += sum(len(s) for s in seqs_y)

        x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

        loss = compute_forward(model=model,
                               critic=critic,
                               seqs_x=x,
                               seqs_y=y,
                               eval=True)

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)

    return float(sum_loss / n_sents)


def bleu_validation(uidx,
                    valid_iterator,
                    model,
                    bleu_scorer,
                    vocab_tgt,
                    batch_size,
                    valid_dir="./valid",
                    max_steps=10,
                    beam_size=5,
                    alpha=-1.0
                    ):
    model.eval()

    numbers = []
    trans = []

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_nums = batch[0]
        numbers += seq_nums

        seqs_x = batch[1]

        infer_progress_bar.update(len(seqs_x))

        x = prepare_data(seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(nmt_model=model, beam_size=beam_size, max_steps=max_steps, src_seqs=x, alpha=alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == EOS:
                    break
                x_tokens.append(vocab_tgt.id2token(wid))

            if len(x_tokens) > 0:
                trans.append(vocab_tgt.tokenizer.detokenize(x_tokens))
            else:
                trans.append('%s' % vocab_tgt.id2token(EOS))

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]

    infer_progress_bar.close()

    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    hyp_path = os.path.join(valid_dir, 'trans.iter{0}.txt'.format(uidx))

    with open(hyp_path, 'w') as f:
        for line in trans:
            f.write('%s\n' % line)

    with open(hyp_path) as f:
        bleu_v = bleu_scorer.corpus_bleu(f)

    return bleu_v


def load_pretrained_model(nmt_model, pretrain_path, device, exclude_prefix=None):
    """
    Args:
        nmt_model: model.
        pretrain_path ('str'): path to pretrained model.
        map_dict ('dict'): mapping specific parameter names to those names
            in current model.
        exclude_prefix ('dict'): excluding parameters with specific names
            for pretraining.

    Raises:
        ValueError: Size not match, parameter name not match or others.

    """
    if exclude_prefix is None:
        exclude_prefix = []
    if pretrain_path != "":
        INFO("Loading pretrained model from {}".format(pretrain_path))
        pretrain_params = torch.load(pretrain_path, map_location=device)
        for name, params in pretrain_params.items():
            flag = False
            for pp in exclude_prefix:
                if name.startswith(pp):
                    flag = True
                    break
            if flag:
                continue
            INFO("Loading param: {}...".format(name))
            try:
                nmt_model.load_state_dict({name: params}, strict=False)
            except Exception as e:
                WARN("{}: {}".format(str(Exception), e))

        INFO("Pretrained model loaded.")


def train(FLAGS):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(FLAGS.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))

    GlobalNames.USE_GPU = FLAGS.use_gpu

    if GlobalNames.USE_GPU:
        CURRENT_DEVICE = "cpu"
    else:
        CURRENT_DEVICE = "cuda:0"

    config_path = os.path.abspath(FLAGS.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    INFO(pretty_configs(configs))

    # Add default configs
    configs = default_configs(configs)
    data_configs = configs['data_configs']

    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    GlobalNames.SEED = training_configs['seed']

    set_seed(GlobalNames.SEED)

    best_model_prefix = os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    train_batch_size = training_configs["batch_size"] * max(1, training_configs["update_cycle"])
    train_buffer_size = training_configs["buffer_size"] * max(1, training_configs["update_cycle"])

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        shuffle=training_configs['shuffle']
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=True)

    bleu_scorer = SacreBLEUScorer(reference_path=data_configs["bleu_valid_reference"],
                                  num_refs=data_configs["num_refs"],
                                  lang_pair=data_configs["lang_pair"],
                                  sacrebleu_args=training_configs["bleu_valid_configs"]['sacrebleu_args'],
                                  postprocess=training_configs["bleu_valid_configs"]['postprocess']
                                  )

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial
    model_collections = Collections()
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(FLAGS.saveto, FLAGS.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    INFO(nmt_model)

    critic = NMTCriterion(label_smoothing=model_configs['label_smoothing'])

    INFO(critic)
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 2. Move to GPU
    if GlobalNames.USE_GPU:
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(nmt_model, FLAGS.pretrain_path, exclude_prefix=None, device=CURRENT_DEVICE)

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=[nmt_model],
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      optim_args=optimizer_configs['optimizer_params']
                      )
    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:

        if optimizer_configs['schedule_method'] == "loss":

            scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                                 **optimizer_configs["scheduler_configs"]
                                                 )

        elif optimizer_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **optimizer_configs['scheduler_configs'])
        else:
            WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(optimizer_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    # 6. build moving average

    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=nmt_model.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if FLAGS.reload:
        checkpoint_saver.load_latest(model=nmt_model, optim=optim, lr_scheduler=scheduler,
                                     collections=model_collections, ma=ma)

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=FLAGS.log_path)

    cum_samples = 0
    cum_words = 0
    best_valid_loss = 1.0 * 1e10  # Max Float
    saving_files = []

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents"
                                     )
        for batch in training_iter:

            uidx += 1

            # if scheduler is None:
            #     pass
            # elif optimizer_configs["schedule_method"] == "loss":
            #     scheduler.step(metric=best_valid_loss)
            # else:
            #     scheduler.step(global_step=uidx)

            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)

            cum_samples += n_samples_t
            cum_words += n_words_t

            training_progress_bar.update(n_samples_t)

            optim.zero_grad()

            try:
                # Prepare data
                for seqs_x_t, seqs_y_t in split_shard(seqs_x, seqs_y, split_size=training_configs['update_cycle']):
                    x, y = prepare_data(seqs_x_t, seqs_y_t, cuda=GlobalNames.USE_GPU)

                    loss = compute_forward(model=nmt_model,
                                           critic=critic,
                                           seqs_x=x,
                                           seqs_y=y,
                                           eval=False,
                                           normalization=n_samples_t,
                                           norm_by_words=training_configs["norm_by_words"])
                optim.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                    optim.zero_grad()
                else:
                    raise e

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=FLAGS.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

                if not is_early_stop:
                    checkpoint_saver.save(global_step=uidx, model=nmt_model, optim=optim, lr_scheduler=scheduler,
                                          collections=model_collections, ma=ma)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=FLAGS.debug):

                if ma is not None:
                    origin_state_dict = deepcopy(nmt_model.state_dict())
                    nmt_model.load_state_dict(ma.export_ma_params(), strict=False)

                valid_loss = loss_validation(model=nmt_model,
                                             critic=critic,
                                             valid_iterator=valid_iterator,
                                             )

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

                best_valid_loss = min_history_loss

                if ma is not None:
                    nmt_model.load_state_dict(origin_state_dict)
                    del origin_state_dict

                if optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=best_valid_loss)

                # ---------------houqi--------------------------------------
                # If model get new best valid loss
                # save model & early stop
                if valid_loss <= best_valid_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(nmt_model.state_dict(), best_model_prefix + ".final")

                        # 2. record all several best models
                        best_model_saver.save(global_step=uidx, model=nmt_model)
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                if ma is not None:
                    nmt_model.load_state_dict(origin_state_dict)
                    del origin_state_dict
                INFO("{0} Loss: {1:.2f}  lrate: {2:6f} patience: {3}".format(
                    uidx, valid_loss, lrate, bad_count
                ))
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break
            # ================================================================================== #
            # BLEU Validation & Early Stop

            # if should_trigger_by_steps(global_step=uidx, n_epoch=eidx,
            #                            every_n_step=training_configs['bleu_valid_freq'],
            #                            min_step=training_configs['bleu_valid_warmup'],
            #                            debug=FLAGS.debug):
            #
            #     if ma is not None:
            #         origin_state_dict = deepcopy(nmt_model.state_dict())
            #         nmt_model.load_state_dict(ma.export_ma_params(), strict=False)
            #
            #     valid_bleu = bleu_validation(uidx=uidx,
            #                                  valid_iterator=valid_iterator,
            #                                  batch_size=training_configs["bleu_valid_batch_size"],
            #                                  model=nmt_model,
            #                                  bleu_scorer=bleu_scorer,
            #                                  vocab_tgt=vocab_tgt,
            #                                  valid_dir=FLAGS.valid_path,
            #                                  max_steps=training_configs["bleu_valid_configs"]["max_steps"],
            #                                  beam_size=training_configs["bleu_valid_configs"]["beam_size"],
            #                                  alpha=training_configs["bleu_valid_configs"]["alpha"]
            #                                  )
            #
            #     model_collections.add_to_collection(key="history_bleus", value=valid_bleu)
            #
            #     best_valid_bleu = float(np.array(model_collections.get_collection("history_bleus")).max())
            #
            #     summary_writer.add_scalar("bleu", valid_bleu, uidx)
            #     summary_writer.add_scalar("best_bleu", best_valid_bleu, uidx)
            #
            #     # If model get new best valid bleu score
            #     if valid_bleu >= best_valid_bleu:
            #         bad_count = 0
            #
            #         if is_early_stop is False:
            #             # 1. save the best model
            #             torch.save(nmt_model.state_dict(), best_model_prefix + ".final")
            #
            #             # 2. record all several best models
            #             best_model_saver.save(global_step=uidx, model=nmt_model)
            #     else:
            #         bad_count += 1
            #
            #         # At least one epoch should be traversed
            #         if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
            #             is_early_stop = True
            #             WARN("Early Stop!")
            #
            #     summary_writer.add_scalar("bad_count", bad_count, uidx)
            #
            #     if ma is not None:
            #         nmt_model.load_state_dict(origin_state_dict)
            #         del origin_state_dict
            #
            #     INFO("{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4}".format(
            #         uidx, valid_loss, valid_bleu, lrate, bad_count
            #     ))


def teacher_translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = ZipDataset(
        TextLineDataset(data_path=FLAGS.source_path,
                        vocabulary=vocab_src),
        TextLineDataset(data_path=FLAGS.target_path,
                        vocabulary=vocab_tgt,
                        ))

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  use_bucket=True, buffer_size=100000,
                                  numbering=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    params = load_model_parameters(FLAGS.model_path, map_location="cpu")

    nmt_model.load_state_dict(params)

    if GlobalNames.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')
    result_numbers = []
    result = []
    n_words = 0

    timer.tic()

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator()
    for batch in valid_iter:

        numbers, seqs_x, seqs_y = batch

        batch_size_t = len(seqs_x)

        x, y = prepare_data(seqs_x=seqs_x, seqs_y=seqs_y, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            probs = nmt_model(x, y, log_probs=False)

        # prob [batch,tgt_len,n_words]
        # max [batch,tgt_len]
        _, word_ids = probs.max(dim=-1)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [wid for wid in sent_t if wid != PAD]
            result.append(sent_t)

            n_words += len(sent_t)

        result_numbers += numbers

        infer_progress_bar.update(batch_size_t)

    infer_progress_bar.close()

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    translation = []
    for sent in result:
        sample = []
        for w in sent:
            if w == vocab_tgt.EOS:
                break
            sample.append(vocab_tgt.id2token(w))

        translation.append(' '.join(sample))

    # resume the ordering
    origin_order = np.argsort(result_numbers).tolist()
    translation = [translation[ii] for ii in origin_order]

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(FLAGS.saveto.split('/')[0:2]))

    with batch_open(FLAGS.saveto, 'w') as handles:
        for trans in translation:
            handles[0].write('%s\n' % trans[6:])


def extract_feature(nmt_model, x, y):
    # x, y = prepare_data(seqs_x=seqs_x, seqs_y=seqs_y, cuda=GlobalNames.USE_GPU)

    # start to extract features
    # y [<BOS> a b c <EOS> <PAD> <PAD>]

    # with torch.no_grad():
    # emb [batch,tgt_len,2*d_model]
    # hidden state [batch,tgt_len,2*d_model]
    # logits vector [batch,tgt_len,n_words]
    # fv [batch,tgt_len,2d_model]

    # emb_feature = nmt_model.extract_emb_feature(y)

    # hidden_feature = nmt_model.extract_hidden_feature(x, y)
    # logit_feature = nmt_model.extract_logit_feature(x, y)
    fv = nmt_model.extract_fv(x, y)

    def process_logit(y, logit_feature):
        # word_id [batch,tgt_len,1]
        word_id = y.detach().unsqueeze(2)

        # word_logit [batch,tgt_len,1]
        word_logit = torch.gather(logit_feature, 2, word_id)

        #  word_logit_max  word_id_max [batch,tgt_len]
        word_logit_max, word_id_max = logit_feature.max(dim=-1)

        #  word_logit_max  word_id_max [batch,tgt_len,1]
        word_logit_max = word_logit_max.unsqueeze(2)
        word_id_max = word_id_max.unsqueeze(2)

        # word_logit-word_logit_max [batch,tgt_len,1]
        word_logit_diff = (word_logit - word_logit_max).type(torch.cuda.FloatTensor)

        # indicator un equal (=1) [batch,tgt_len,1]
        word_id_ind = (1 - word_id.eq(word_id_max)).type(torch.cuda.FloatTensor)

        # concat [batch,tgt_len,4]
        logit_feature = torch.cat((word_logit, word_logit_max,
                                   word_logit_diff, word_id_ind), 2)

        return logit_feature

    # logit_feature [batch,tgt_len,4]

    # logit_feature = process_logit(y, logit_feature)

    # return logit_feature
    # all_feature [batch,tgt_len,4*d_model+4]

    # all_feature = torch.cat((emb_feature, hidden_feature,
    #                          logit_feature), 2)

    # all_feature = torch.cat(( hidden_feature,emb_feature), 2)

    # all_feature = torch.cat(( fv,emb_feature), 2)

    # all_feature = torch.cat((fv, logit_feature), 2)
    # all_feature = emb_feature
    # all_feature = hidden_feature
    all_feature = fv

    return all_feature

def load_label_data(data_path):
    fy = open(data_path, 'r')
    array = fy.readlines()
    listy = []

    for line in array:
        line = np.float32(line.strip())
        listy.append(line)
    return np.array(listy)


def train_rnn(FLAGS):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(FLAGS.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))

    GlobalNames.USE_GPU = FLAGS.use_gpu

    if GlobalNames.USE_GPU:
        CURRENT_DEVICE = "cpu"
    else:
        CURRENT_DEVICE = "cuda:0"

    config_path = os.path.abspath(FLAGS.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    INFO(pretty_configs(configs))

    # Add default configs
    configs = default_configs(configs)

    transformer_model_configs = configs['transformer_model_configs']

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    GlobalNames.SEED = training_configs['seed']

    set_seed(GlobalNames.SEED)

    best_model_prefix = os.path.join(FLAGS.saveto_transformer_model,
                                     FLAGS.transformer_model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)
    best_model_prefix_qe = os.path.join(FLAGS.saveto_qe_model, FLAGS.qe_model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    # train_batch_size = training_configs["batch_size"] * max(1, training_configs["update_cycle"])
    # train_buffer_size = training_configs["buffer_size"] * max(1, training_configs["update_cycle"])
    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        shuffle=training_configs['shuffle']
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    # loading train label data
    train_label = load_label_data(data_path=data_configs['train_data'][2])

    # loading valid label data
    valid_label = load_label_data(data_path=data_configs['valid_data'][2])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(FLAGS.saveto_transformer_model,
                                                                        FLAGS.transformer_model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    checkpoint_saver_qe = Saver(save_prefix="{0}.ckpt".format(os.path.join(FLAGS.saveto_qe_model, FLAGS.qe_model_name)),
                                num_max_keeping=training_configs['num_kept_checkpoints']
                                )
    best_model_saver_qe = Saver(save_prefix=best_model_prefix_qe,
                                num_max_keeping=training_configs['num_kept_best_model'])

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **transformer_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    params = load_model_parameters(FLAGS.model_path, map_location="cpu")
    nmt_model.load_state_dict(params)
    if GlobalNames.USE_GPU:
        nmt_model.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 1. load Bi-LSTM Model
    INFO('Building QE model...')
    timer.tic()
    qe_model = build_model(**model_configs)
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # INFO('Reloading QE model parameters...')
    # timer.tic()
    # qe_params = load_model_parameters(FLAGS.qe_model_path, map_location="cpu")
    # qe_model.load_state_dict(qe_params)

    critic = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 2. Move to GPU
    if GlobalNames.USE_GPU:
        qe_model = qe_model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(qe_model, FLAGS.pretrain_path, exclude_prefix=None, device=CURRENT_DEVICE)

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=[nmt_model, qe_model],
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:

        if optimizer_configs['schedule_method'] == "loss":

            scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                                 **optimizer_configs["scheduler_configs"]
                                                 )

        elif optimizer_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **optimizer_configs['scheduler_configs'])
        else:
            WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(optimizer_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    # if FLAGS.reload:
    #     checkpoint_saver.load_latest(model=nmt_model, optim=optim, lr_scheduler=scheduler,
    #                                  collections=model_collections, ma=ma)
    #     checkpoint_saver_qe.load_latest(model=qe_model, optim=optim, lr_scheduler=scheduler,
    #                                     collections=model_collections, ma=ma)

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=FLAGS.log_path)

    cum_samples = 0
    cum_words = 0
    best_valid_loss = 1.0 * 1e10  # Max Float

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        _end = 0
        for batch in training_iter:
            # nmt_model.train()
            # nmt_model.decoder.dropout.eval()
            nmt_model.eval()
            qe_model.train()

            uidx += 1

            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)

            cum_samples += n_samples_t
            cum_words += n_words_t

            training_progress_bar.update(n_samples_t)

            # optim.zero_grad()

            _start = _end
            _end += n_samples_t

            train_loss = 0
            # try:
            # for seqs_x_t, seqs_y_t in split_shard(seqs_x, seqs_y, split_size=training_configs['update_cycle']):
            # extract feature using transformer
            x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

            # loading train golden label
            # _start = _end
            # _end += len(seqs_x_t)

            xy_label = train_label[_start:_end]
            xy_label = torch.tensor(xy_label).unsqueeze(1)
            if GlobalNames.USE_GPU:
                xy_label = xy_label.cuda()
            # with torch.no_grad():
            #     # all_feature [batch,tgt_len,4*d_model*4]
            #     all_feature = extract_feature(nmt_model, x, y)
            #     all_feature = all_feature.detach()
            with torch.enable_grad():
                # all_feature [batch,tgt_len,4*d_model*4]
                all_feature = extract_feature(nmt_model, x, y)

                loss = compute_forward_qe(model=qe_model,
                                          critic=critic,
                                          feature=all_feature,
                                          y=y,
                                          y_label=xy_label,
                                          eval=False)
            # --------------------------------------------
            if training_configs['update_cycle'] > 1:
                loss = loss / training_configs['update_cycle']

            torch.autograd.backward(loss)
            train_loss += loss.item()
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            # except RuntimeError as e:
            #     if 'out of memory' in str(e):
            #         print('| WARNING: ran out of memory, skipping batch')
            #         oom_count += 1
            #         optim.zero_grad()
            #     else:
            #         raise e

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} ".format(
                    uidx, train_loss
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=FLAGS.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

                # if not is_early_stop:
                #     checkpoint_saver.save(global_step=uidx, model=nmt_model, optim=optim, lr_scheduler=scheduler,
                #                           collections=model_collections, ma=ma)
                #     checkpoint_saver_qe.save(global_step=uidx, model=qe_model, optim=optim, lr_scheduler=scheduler,
                #                              collections=model_collections, ma=ma)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=FLAGS.debug):

                if ma is not None:
                    origin_state_dict = deepcopy(qe_model.state_dict())
                    nmt_model.load_state_dict(ma.export_ma_params(), strict=False)

                n_tokens = 0.0
                tol_sents = 0
                sum_loss = 0.0

                valid_iter = valid_iterator.build_generator()

                _end_valid = 0
                nb_eval_steps = 0

                nmt_model.eval()
                qe_model.eval()

                for batch_valid in valid_iter:
                    seqs_x_valid, seqs_y_valid = batch_valid

                    n_sents = len(seqs_x_valid)

                    n_tokens += sum(len(s) for s in seqs_y_valid)
                    tol_sents += n_sents

                    _start_valid = _end_valid
                    _end_valid += n_sents

                    # extract feature using transformer
                    x_valid, y_valid = prepare_data(seqs_x_valid, seqs_y_valid, cuda=GlobalNames.USE_GPU)

                    # loading train golden label
                    xy_label_valid = valid_label[_start_valid:_end_valid]
                    xy_label_valid = torch.tensor(xy_label_valid).unsqueeze(1)
                    if GlobalNames.USE_GPU:
                        xy_label_valid = xy_label_valid.cuda()

                    with torch.no_grad():
                        # all_feature [batch,tgt_len,4*d_model+4]
                        all_feature_valid = extract_feature(nmt_model, x_valid, y_valid)
                        all_feature_valid_detach = all_feature_valid.detach()

                        valid_loss_batch = compute_forward_qe(model=qe_model,
                                                              critic=critic,
                                                              feature=all_feature_valid_detach,
                                                              y=y_valid,
                                                              y_label=xy_label_valid,
                                                              eval=True)
                    if np.isnan(valid_loss_batch):
                        WARN("NaN detected!")

                    sum_loss += float(valid_loss_batch)
                    nb_eval_steps += 1

                valid_loss = float(sum_loss / nb_eval_steps)

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

                best_valid_loss = min_history_loss

                if ma is not None:
                    nmt_model.load_state_dict(origin_state_dict)
                    del origin_state_dict

                if optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=best_valid_loss)

                # ---------------houqi--------------------------------------
                # If model get new best valid loss
                # save model & early stop
                if valid_loss <= best_valid_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(nmt_model.state_dict(), best_model_prefix + ".final")
                        torch.save(qe_model.state_dict(), best_model_prefix_qe + ".final")

                        # 2. record all several best models
                        best_model_saver.save(global_step=uidx, model=nmt_model)
                        best_model_saver_qe.save(global_step=uidx, model=qe_model)
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                if ma is not None:
                    nmt_model.load_state_dict(origin_state_dict)
                    del origin_state_dict

                INFO("{0} Loss: {1:.8f}  lrate: {2:6f} patience: {3}".format(
                    uidx, valid_loss, lrate, bad_count
                ))
            if is_early_stop == True:
                break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break

from torch.nn.parameter import Parameter
def train_ensemble():

    train_batch_size = 32
    ensemble_num=10
    best_model_prefix = os.path.join("./","fc.best")

    train_bitext_dataset = ZipDataset(
        QEDataset(data_path="./en2de2019_result/train-10.txt"), shuffle=False)

    valid_bitext_dataset = ZipDataset(
        QEDataset(data_path="./en2de2019_result/dev-10.txt"), shuffle=False)

    training_iterator = DataIterator(dataset=train_bitext_dataset, batch_size=train_batch_size)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset, batch_size=train_batch_size)

    train_label = load_label_data("./en2de2019_result/train.hter")

    valid_label = load_label_data("./en2de2019_result/dev.hter")

    model_collections = Collections()
    best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=1)

    full_connect = torch.nn.Linear(ensemble_num, 1, bias=False)
    torch.nn.init.xavier_normal_(full_connect.weight)

    critic = torch.nn.MSELoss(reduce=True, size_average=True)

    full_connect = full_connect.cuda()
    critic = critic.cuda()

    is_early_stop = False
    patience=20
    bad_count = 0
    optimizer = "adagrad"
    lrate = 0.0001
    grad_clip = 1.0
    optim = Optimizer(name=optimizer, model=[full_connect], lr=lrate,
                      grad_clip=grad_clip, optim_args=None)
    uidx = 0
    eidx=0
    while True:
        training_iter = training_iterator.build_generator()
        _end = 0

        for batch in training_iter:

            full_connect.train()
            uidx += 1

            seqs_x = batch
            seqs_x=torch.tensor(seqs_x)
            n_samples_t=len(seqs_x)
            # print("seq_x",seqs_x)

            optim.zero_grad()

            _start = _end
            _end += n_samples_t

            xy_label = train_label[_start:_end]
            xy_label = torch.tensor(xy_label).unsqueeze(1)

            seqs_x = seqs_x.cuda()
            xy_label = xy_label.cuda()

            full_connect.weight=Parameter(torch.nn.functional.softmax(full_connect.weight))
            # print("xy_label", xy_label)
            # temp = full_connect.weight.data.cpu().numpy() / sum(full_connect.weight.data.cpu().numpy())
            # temp = torch.tensor(temp)
            # temp = torch.tensor(temp).cuda()
            # full_connect.weight = Parameter(temp)
            with torch.enable_grad():
                logits = full_connect(seqs_x)

                loss = critic(logits.view(-1), xy_label.view(-1))



            torch.autograd.backward(loss)
            optim.step()


            if uidx%100==0:
                INFO("{0} TrainLoss: {1:.4f} ".format( uidx, loss ))

            if uidx%500==0:
                print("weight", full_connect.weight.data)
                sum_loss = 0.0
                nb_eval_steps = 0

                valid_iter = valid_iterator.build_generator()
                _end_valid = 0

                full_connect.eval()

                for batch_valid in valid_iter:
                    seqs_x_valid= batch_valid
                    n_sents = len(seqs_x_valid)
                    seqs_x_valid = torch.tensor(seqs_x_valid)

                    _start_valid = _end_valid
                    _end_valid += n_sents

                    xy_label_valid = valid_label[_start_valid:_end_valid]
                    xy_label_valid = torch.tensor(xy_label_valid).unsqueeze(1)

                    seqs_x_valid=seqs_x_valid.cuda()
                    xy_label_valid = xy_label_valid.cuda()

                    with torch.no_grad():
                        logits = full_connect(seqs_x_valid)
                        valid_loss_batch = critic(logits.view(-1), xy_label_valid.view(-1))

                    if np.isnan(valid_loss_batch):
                        WARN("NaN detected!")

                    sum_loss += float(valid_loss_batch)
                    nb_eval_steps += 1

                valid_loss = float(sum_loss / nb_eval_steps)

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()
                best_valid_loss = min_history_loss

                # ---------------houqi--------------------------------------
                # If model get new best valid loss
                # save model & early stop
                if valid_loss <= best_valid_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(full_connect.state_dict(), best_model_prefix + ".final")

                        best_model_saver.save(global_step=uidx, model=full_connect)

                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= patience and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                INFO("{0} Loss: {1:.8f}  lrate: {2:6f} patience: {3}".format(
                    uidx, valid_loss, lrate, bad_count
                ))
            if is_early_stop == True:
                break
        eidx += 1
        if eidx > 100000:
            break


def test_rnn(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    transformer_model_configs = configs['transformer_model_configs']

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = ZipDataset(
        TextLineDataset(data_path=FLAGS.source_path,
                        vocabulary=vocab_src),
        TextLineDataset(data_path=FLAGS.target_path,
                        vocabulary=vocab_tgt,
                        ))

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **transformer_model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(FLAGS.transformer_model_path, map_location="cpu")
    nmt_model.load_state_dict(params)
    if GlobalNames.USE_GPU:
        nmt_model.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 1. load Bi-LSTM Model
    INFO('Building QE model...')
    timer.tic()
    qe_model = build_model(**model_configs)
    qe_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading QE model parameters...')
    timer.tic()
    qe_params = load_model_parameters(FLAGS.qe_model_path, map_location="cpu")
    qe_model.load_state_dict(qe_params)
    if GlobalNames.USE_GPU:
        qe_model.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')
    result = []
    timer.tic()

    valid_iter = valid_iterator.build_generator()

    for batch_valid in valid_iter:
        seqs_x_valid, seqs_y_valid = batch_valid

        # extract feature using transformer
        x_valid, y_valid = prepare_data(seqs_x_valid, seqs_y_valid, cuda=GlobalNames.USE_GPU)

        # predict label
        # outputs_valid [batch,1]
        with torch.no_grad():
            # all_feature [batch,tgt_len,4*d_model*4]
            all_feature_valid = extract_feature(nmt_model, x_valid, y_valid)
            all_feature_valid_detach = all_feature_valid.detach()

            outputs_valid = qe_model(all_feature_valid_detach, y_valid)
        # outputs_valid = torch.clamp(outputs_valid, 0, 1)
        outputs_valid.squeeze(1).cpu().numpy().tolist()

        for val in outputs_valid:
            result.append(float(val))

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(FLAGS.saveto.split('/')[0:2]))
    with open(FLAGS.saveto, 'w') as f:
        for val in result:
            f.write('%.6f\n' % val)


def train_rnn_bt(FLAGS):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(FLAGS.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))

    GlobalNames.USE_GPU = FLAGS.use_gpu

    if GlobalNames.USE_GPU:
        CURRENT_DEVICE = "cpu"
    else:
        CURRENT_DEVICE = "cuda:0"

    config_path = os.path.abspath(FLAGS.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    INFO(pretty_configs(configs))

    # Add default configs
    configs = default_configs(configs)

    transformer_model_configs = configs['transformer_model_configs']
    transformer_model_bt_configs = configs['transformer_model_bt_configs']

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    GlobalNames.SEED = training_configs['seed']

    set_seed(GlobalNames.SEED)

    best_model_prefix = os.path.join(FLAGS.saveto_transformer_model,
                                     FLAGS.transformer_model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)
    best_model_prefix_bt = os.path.join(FLAGS.saveto_transformer_model_bt,
                                        FLAGS.transformer_model_bt_name + GlobalNames.MY_BEST_MODEL_SUFFIX)

    best_model_prefix_qe = os.path.join(FLAGS.saveto_qe_model, FLAGS.qe_model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    # train_batch_size = training_configs["batch_size"] * max(1, training_configs["update_cycle"])
    # train_buffer_size = training_configs["buffer_size"] * max(1, training_configs["update_cycle"])
    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        shuffle=training_configs['shuffle']
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    # loading train label data
    train_label = load_label_data(data_path=data_configs['train_data'][2])

    # loading valid label data
    valid_label = load_label_data(data_path=data_configs['valid_data'][2])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(FLAGS.saveto_transformer_model,
                                                                        FLAGS.transformer_model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    checkpoint_saver_bt = Saver(save_prefix="{0}.ckpt".format(os.path.join(FLAGS.saveto_transformer_model_bt,
                                                                           FLAGS.transformer_model_bt_name)),
                                num_max_keeping=training_configs['num_kept_checkpoints']
                                )
    best_model_saver_bt = Saver(save_prefix=best_model_prefix_bt,
                                num_max_keeping=training_configs['num_kept_best_model'])

    checkpoint_saver_qe = Saver(save_prefix="{0}.ckpt".format(os.path.join(FLAGS.saveto_qe_model, FLAGS.qe_model_name)),
                                num_max_keeping=training_configs['num_kept_checkpoints']
                                )
    best_model_saver_qe = Saver(save_prefix=best_model_prefix_qe,
                                num_max_keeping=training_configs['num_kept_best_model'])

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **transformer_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    params = load_model_parameters(FLAGS.model_path, map_location="cpu")
    nmt_model.load_state_dict(params)
    if GlobalNames.USE_GPU:
        nmt_model.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0.0. load transformer_bt
    INFO('Building transformer_bt model...')
    timer.tic()
    nmt_model_bt = build_model(n_src_vocab=vocab_tgt.max_n_words,
                               n_tgt_vocab=vocab_src.max_n_words, **transformer_model_bt_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model_bt parameters...')
    timer.tic()
    params_bt = load_model_parameters(FLAGS.model_bt_path, map_location="cpu")
    nmt_model_bt.load_state_dict(params_bt)
    if GlobalNames.USE_GPU:
        nmt_model_bt.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 1. Build Bi-LSTM Model & Criterion
    INFO('Building QE model...')
    timer.tic()

    qe_model = build_model(**model_configs)
    INFO(qe_model)

    # INFO('Reloading QE model parameters...')
    # timer.tic()
    # qe_params = load_model_parameters(FLAGS.qe_model_path, map_location="cpu")
    # qe_model.load_state_dict(qe_params)

    critic = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 2. Move to GPU
    if GlobalNames.USE_GPU:
        qe_model = qe_model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(qe_model, FLAGS.pretrain_path, exclude_prefix=None, device=CURRENT_DEVICE)

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=[nmt_model, nmt_model_bt, qe_model],
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )
    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:

        if optimizer_configs['schedule_method'] == "loss":

            scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                                 **optimizer_configs["scheduler_configs"]
                                                 )

        elif optimizer_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **optimizer_configs['scheduler_configs'])
        else:
            WARN("Unknown scheduler name {0}. Do not use lr_scheduling.".format(optimizer_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    # if FLAGS.reload:
    #     checkpoint_saver.load_latest(model=nmt_model, optim=optim, lr_scheduler=scheduler,
    #                                  collections=model_collections, ma=ma)
    #     checkpoint_saver_bt.load_latest(model=nmt_model_bt, optim=optim, lr_scheduler=scheduler,
    #                                     collections=model_collections, ma=ma)
    #     checkpoint_saver_qe.load_latest(model=qe_model, optim=optim, lr_scheduler=scheduler,
    #                                     collections=model_collections, ma=ma)

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=FLAGS.log_path)

    cum_samples = 0
    cum_words = 0
    best_valid_loss = 1.0 * 1e10  # Max Float

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        _end = 0
        for batch in training_iter:

            nmt_model.eval()

            nmt_model_bt.eval()

            qe_model.train()

            uidx += 1

            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)

            cum_samples += n_samples_t
            cum_words += n_words_t

            training_progress_bar.update(n_samples_t)

            # optim.zero_grad()

            _start = _end
            _end += n_samples_t

            # try:
            # extract feature using transformer
            train_loss = 0
            # for seqs_x_t, seqs_y_t in split_shard(seqs_x, seqs_y, split_size=training_configs['update_cycle']):
            x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

            # _start = _end
            # _end += len(x)

            # loading train golden label
            xy_label = train_label[_start:_end]
            # print(xy_label)
            xy_label = torch.tensor(xy_label).unsqueeze(1)

            if GlobalNames.USE_GPU:
                xy_label = xy_label.cuda()
            #
            # with torch.no_grad():
            #     # all_feature [batch,tgt_len,4*d_model*4]
            #     all_feature = extract_feature(nmt_model, x, y)
            #
            #     # extract feature using transformer_bt
            #     # all_feature_bt [batch,src_len,4*d_model*4]
            #     all_feature_bt = extract_feature(nmt_model_bt, y, x)

            with torch.enable_grad():
                all_feature = extract_feature(nmt_model, x, y)

                # extract feature using transformer_bt
                # all_feature_bt [batch,src_len,4*d_model*4]
                all_feature_bt = extract_feature(nmt_model_bt, y, x)

                loss = compute_forward_qe_bt(model=qe_model,
                                             critic=critic,
                                             feature=all_feature,
                                             feature_bt=all_feature_bt,
                                             y=y,
                                             x=x,
                                             y_label=xy_label,
                                             eval=False)
            # -------------------------------------------------------------
            if training_configs['update_cycle'] > 1:
                loss = loss / training_configs['update_cycle']
            loss.backward()
            train_loss += loss.item()

            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()


                # -------------------------------------------------------------

                # optim.step()
                # print("dddd")
            # except RuntimeError as e:
            #     if 'out of memory' in str(e):
            #         print('| WARNING: ran out of memory, skipping batch')
            #         oom_count += 1
            #         optim.zero_grad()
            #     else:
            #         raise e
            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()
            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]
                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} ".format(
                    uidx, train_loss
                ))
                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=FLAGS.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)
                # if not is_early_stop:
                #     checkpoint_saver.save(global_step=uidx, model=nmt_model, optim=optim, lr_scheduler=scheduler,
                #                           collections=model_collections, ma=ma)
                #     checkpoint_saver_bt.save(global_step=uidx, model=nmt_model_bt, optim=optim, lr_scheduler=scheduler,
                #                           collections=model_collections, ma=ma)
                #     checkpoint_saver_qe.save(global_step=uidx, model=qe_model, optim=optim, lr_scheduler=scheduler,
                #                              collections=model_collections, ma=ma)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=FLAGS.debug):

                if ma is not None:
                    origin_state_dict = deepcopy(qe_model.state_dict())
                    nmt_model.load_state_dict(ma.export_ma_params(), strict=False)

                n_tokens = 0.0
                tol_sents = 0
                sum_loss = 0.0

                valid_iter = valid_iterator.build_generator()
                _end_valid = 0
                nb_eval_steps = 0

                nmt_model.eval()
                nmt_model_bt.eval()
                qe_model.eval()

                for batch_valid in valid_iter:
                    seqs_x_valid, seqs_y_valid = batch_valid

                    n_sents = len(seqs_x_valid)

                    n_tokens += sum(len(s) for s in seqs_y_valid)
                    tol_sents += n_sents

                    _start_valid = _end_valid
                    _end_valid += n_sents

                    # extract feature using transformer
                    x_valid, y_valid = prepare_data(seqs_x_valid, seqs_y_valid, cuda=GlobalNames.USE_GPU)

                    # loading train golden label
                    xy_label_valid = valid_label[_start_valid:_end_valid]
                    xy_label_valid = torch.tensor(xy_label_valid).unsqueeze(1)
                    if GlobalNames.USE_GPU:
                        xy_label_valid = xy_label_valid.cuda()

                    with torch.no_grad():
                        # all_feature [batch,tgt_len,4*d_model+4]
                        all_feature_valid = extract_feature(nmt_model, x_valid, y_valid)

                        # extract feature using transformer_bt
                        # all_feature_bt [batch,src_len,4*d_model+4]
                        all_feature_valid_bt = extract_feature(nmt_model_bt, y_valid, x_valid)

                        valid_loss_batch = compute_forward_qe_bt(model=qe_model,
                                                                 critic=critic,
                                                                 feature=all_feature_valid,
                                                                 feature_bt=all_feature_valid_bt,
                                                                 y=y_valid,
                                                                 x=x_valid,
                                                                 y_label=xy_label_valid,
                                                                 eval=True)
                    if np.isnan(valid_loss_batch):
                        WARN("NaN detected!")

                    sum_loss += float(valid_loss_batch)
                    nb_eval_steps += 1

                valid_loss = float(sum_loss / nb_eval_steps)

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

                best_valid_loss = min_history_loss

                if ma is not None:
                    nmt_model.load_state_dict(origin_state_dict)
                    del origin_state_dict

                if optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=best_valid_loss)

                # ---------------houqi--------------------------------------
                # If model get new best valid loss
                # save model & early stop
                if valid_loss <= best_valid_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(nmt_model.state_dict(), best_model_prefix + ".final")
                        torch.save(nmt_model_bt.state_dict(), best_model_prefix_bt + ".final")
                        torch.save(qe_model.state_dict(), best_model_prefix_qe + ".final")

                        # 2. record all several best models
                        best_model_saver.save(global_step=uidx, model=nmt_model)
                        best_model_saver_bt.save(global_step=uidx, model=nmt_model_bt)
                        best_model_saver_qe.save(global_step=uidx, model=qe_model)
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                if ma is not None:
                    nmt_model.load_state_dict(origin_state_dict)
                    del origin_state_dict

                INFO("{0} ValidLoss: {1:.4f}  lrate: {2:6f} patience: {3}".format(
                    uidx, valid_loss, lrate, bad_count
                ))
            if is_early_stop == True:
                break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def test_rnn_bt(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    transformer_model_configs = configs['transformer_model_configs']
    transformer_model_bt_configs = configs['transformer_model_bt_configs']

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = ZipDataset(
        TextLineDataset(data_path=FLAGS.source_path,
                        vocabulary=vocab_src),
        TextLineDataset(data_path=FLAGS.target_path,
                        vocabulary=vocab_tgt,
                        ))

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **transformer_model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(FLAGS.transformer_model_path, map_location="cpu")
    nmt_model.load_state_dict(params)
    if GlobalNames.USE_GPU:
        nmt_model.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0.0. load transformer_bt
    INFO('Building transformer_bt model...')
    timer.tic()
    nmt_model_bt = build_model(n_src_vocab=vocab_tgt.max_n_words,
                               n_tgt_vocab=vocab_src.max_n_words, **transformer_model_bt_configs)
    nmt_model_bt.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model_bt parameters...')
    timer.tic()
    params_bt = load_model_parameters(FLAGS.transformer_model_bt_path, map_location="cpu")
    nmt_model_bt.load_state_dict(params_bt)
    if GlobalNames.USE_GPU:
        nmt_model_bt.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 1. load Bi-LSTM Model
    INFO('Building QE model...')
    timer.tic()
    qe_model = build_model(**model_configs)
    qe_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading QE model parameters...')
    timer.tic()
    qe_params = load_model_parameters(FLAGS.qe_model_path, map_location="cpu")
    qe_model.load_state_dict(qe_params)
    if GlobalNames.USE_GPU:
        qe_model.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')
    result = []
    timer.tic()

    valid_iter = valid_iterator.build_generator()

    for batch_valid in valid_iter:
        seqs_x_valid, seqs_y_valid = batch_valid

        # extract feature using transformer
        x_valid, y_valid = prepare_data(seqs_x_valid, seqs_y_valid, cuda=GlobalNames.USE_GPU)

        # predict label
        # outputs_valid [batch,1]
        with torch.no_grad():
            # all_feature [batch,tgt_len,4*d_model+4]
            all_feature_valid = extract_feature(nmt_model, x_valid, y_valid)

            # extract feature using transformer_bt
            # all_feature [batch,tgt_len,4*d_model+4]
            all_feature_valid_bt = extract_feature(nmt_model_bt, y_valid, x_valid)
            outputs_valid = qe_model(all_feature_valid, y_valid, all_feature_valid_bt, x_valid)
        # outputs_valid = torch.clamp(outputs_valid, 0, 1)
        outputs_valid.squeeze(1).cpu().numpy().tolist()

        for val in outputs_valid:
            result.append(float(val))

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(FLAGS.saveto.split('/')[0:2]))
    with open(FLAGS.saveto, 'w') as f:
        for val in result:
            f.write('%.6f\n' % val)


def translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = TextLineDataset(data_path=FLAGS.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  use_bucket=True, buffer_size=100000, numbering=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()
    nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                            n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    nmt_model.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()

    params = load_model_parameters(FLAGS.model_path, map_location="cpu")

    nmt_model.load_state_dict(params)

    if GlobalNames.USE_GPU:
        nmt_model.cuda()

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')

    result_numbers = []
    result = []
    n_words = 0

    timer.tic()

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator()
    for batch in valid_iter:

        numbers, seqs_x = batch

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = beam_search(nmt_model=nmt_model, beam_size=FLAGS.beam_size, max_steps=FLAGS.max_steps,
                                   src_seqs=x, alpha=FLAGS.alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            result.append(sent_t)

            n_words += len(sent_t[0])

        result_numbers += numbers

        infer_progress_bar.update(batch_size_t)

    infer_progress_bar.close()

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    translation = []
    for sent in result:
        samples = []
        for trans in sent:
            sample = []
            for w in trans:
                if w == vocab_tgt.EOS:
                    break
                sample.append(vocab_tgt.id2token(w))
            samples.append(vocab_tgt.tokenizer.detokenize(sample))
        translation.append(samples)

    # resume the ordering
    origin_order = np.argsort(result_numbers).tolist()
    translation = [translation[ii] for ii in origin_order]

    keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
    outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

    with batch_open(outputs, 'w') as handles:
        for trans in translation:
            for i in range(keep_n):
                if i < len(trans):
                    handles[i].write('%s\n' % trans[i])
                else:
                    handles[i].write('%s\n' % 'eos')


def ensemble_translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = TextLineDataset(data_path=FLAGS.source_path,
                                    vocabulary=vocab_src)

    valid_iterator = DataIterator(dataset=valid_dataset,
                                  batch_size=FLAGS.batch_size,
                                  use_bucket=True, buffer_size=100000, numbering=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    nmt_models = []

    model_path = FLAGS.model_path

    for ii in range(len(model_path)):

        nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
        nmt_model.eval()
        INFO('Done. Elapsed time {0}'.format(timer.toc()))

        INFO('Reloading model parameters...')
        timer.tic()

        params = load_model_parameters(model_path[ii], map_location="cpu")

        nmt_model.load_state_dict(params)

        if GlobalNames.USE_GPU:
            nmt_model.cuda()

        nmt_models.append(nmt_model)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')
    result_numbers = []
    result = []
    n_words = 0

    timer.tic()

    infer_progress_bar = tqdm(total=len(valid_iterator),
                              desc=' - (Infer)  ',
                              unit="sents")

    valid_iter = valid_iterator.build_generator()
    for batch in valid_iter:

        numbers, seqs_x = batch

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = ensemble_beam_search(nmt_models=nmt_models, beam_size=FLAGS.beam_size, max_steps=FLAGS.max_steps,
                                            src_seqs=x, alpha=FLAGS.alpha)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            result.append(sent_t)

            n_words += len(sent_t[0])

        infer_progress_bar.update(batch_size_t)

    infer_progress_bar.close()

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    translation = []
    for sent in result:
        samples = []
        for trans in sent:
            sample = []
            for w in trans:
                if w == vocab_tgt.EOS:
                    break
                sample.append(vocab_tgt.id2token(w))
            samples.append(vocab_tgt.tokenizer.detokenize(sample))
        translation.append(samples)

    # resume the ordering
    origin_order = np.argsort(result_numbers).tolist()
    translation = [translation[ii] for ii in origin_order]

    keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
    outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

    with batch_open(outputs, 'w') as handles:
        for trans in translation:
            for i in range(keep_n):
                if i < len(trans):
                    handles[i].write('%s\n' % trans[i])
                else:
                    handles[i].write('%s\n' % 'eos')
