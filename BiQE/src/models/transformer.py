# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import PAD
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.modules.basic import BottleLinear as Linear
from src.modules.embeddings import Embeddings
from src.modules.sublayers import PositionwiseFeedForward, MultiHeadedAttention
from src.utils import nest
from src.utils.common_utils import *
from src.utils.logging import *


def get_attn_causal_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    '''
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)


class Encoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, dim_per_head=None):
        super().__init__()

        self.num_layers = n_layers
        self.embeddings = Embeddings(num_embeddings=n_src_vocab,
                                     embedding_dim=d_word_vec,
                                     dropout=dropout,
                                     add_position_embedding=True
                                     )
        self.block_stack = nn.ModuleList(
            [EncoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout,
                          dim_per_head=dim_per_head)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_seq):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.embeddings(src_seq)

        enc_mask = src_seq.detach().eq(PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = emb

        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)

        out = self.layer_norm(out)

        return out, enc_mask


class DecoderBlock(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.ctx_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def compute_cache(self, enc_output):
        return self.ctx_attn.compute_cache(enc_output, enc_output)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        input_norm = self.layer_norm_1(dec_input)
        all_input = input_norm

        query, _, self_attn_cache = self.slf_attn(all_input, all_input, input_norm,
                                                  mask=slf_attn_mask, self_attn_cache=self_attn_cache)

        query = self.dropout(query) + dec_input

        query_norm = self.layer_norm_2(query)
        mid, attn, enc_attn_cache = self.ctx_attn(enc_output, enc_output, query_norm,
                                                  mask=dec_enc_attn_mask, enc_attn_cache=enc_attn_cache)

        output = self.pos_ffn(self.dropout(mid) + query)

        return output, attn, self_attn_cache, enc_attn_cache


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None, dropout=0.1):

        super(Decoder, self).__init__()

        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model

        self.embeddings = Embeddings(n_tgt_vocab, d_word_vec,
                                     dropout=dropout, add_position_embedding=True)

        self.block_stack = nn.ModuleList([
                                             DecoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head,
                                                          dropout=dropout,
                                                          dim_per_head=dim_per_head)
                                             for _ in range(n_layers)])

        self.out_layer_norm = nn.LayerNorm(d_model)

        self._dim_per_head = dim_per_head

    @property
    def dim_per_head(self):
        if self._dim_per_head is None:
            return self.d_model // self.n_head
        else:
            return self._dim_per_head

    def forward(self, tgt_seq, enc_output, enc_mask, enc_attn_caches=None, self_attn_caches=None):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_seq)

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(PAD).unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = emb
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache \
                = self.block_stack[i](output,
                                      enc_output,
                                      dec_slf_attn_mask,
                                      dec_enc_attn_mask,
                                      enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                                      self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None)

            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        output = self.out_layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches


class Generator(nn.Module):
    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):
        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

    def _pad_2d(self, x):

        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True):
        """
        input == > Linear == > LogSoftmax
        """

        logits = self.proj(input)

        logits = self._pad_2d(logits)

        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def cul_logits(self, input):

        logits = self.proj(input)

        logits = self._pad_2d(logits)

        return logits


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None,
            dropout=0.1, proj_share_weight=False, **kwargs):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            n_src_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head)

        self.decoder = Decoder(
            n_tgt_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head)

        self.decoder_backward = Decoder(
            n_tgt_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head)

        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        if proj_share_weight:
            self.generator = Generator(n_words=n_tgt_vocab,
                                       hidden_size=2 * d_word_vec,
                                       shared_weight=self.decoder.embeddings.embeddings.weight,
                                       padding_idx=PAD)

        else:
            self.generator = Generator(n_words=n_tgt_vocab, hidden_size=2 * d_word_vec, padding_idx=PAD)

        # -----------------houq------------------------------------------------------------------------
        self.out_layer_norm = nn.LayerNorm(2 * d_model)
        # -----------------------------------new-------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        self.decoder.embeddings.embeddings.weight = self.decoder_backward.embeddings.embeddings.weight

    # ------------------------houq--------------------------------------------
    def forward(self, src_seq, tgt_seq, log_probs=True):

        enc_output, enc_mask = self.encoder(src_seq)

        tgt_seq_forward = tgt_seq[:, :-1].contiguous()

        dec_output, _, _ = self.decoder(tgt_seq_forward, enc_output, enc_mask)

        # add backward self-attention
        tgt_seq_reverse = []

        for i in tgt_seq:
            tgt_seq_reverse.append(list(reversed(i.data.cpu().numpy())))

        tgt_seq_reverse = torch.tensor(np.array(tgt_seq_reverse))

        tgt_seq_reverse = tensorToCuda(tgt_seq_reverse)

        tgt_seq_backward = tgt_seq_reverse[:, :-1].contiguous()

        # tgt_seq_backward=tgt_seq[:,:0:-1].contiguous()
        dec_output_backward, _, _ = self.decoder_backward(tgt_seq_backward, enc_output, enc_mask)

        # padding zero

        batch_size, tgt_len, dim = dec_output.size()

        padding_zero = torch.zeros(batch_size, 1, dim)

        padding_zero = tensorToCuda(padding_zero)

        dec_output_zero = torch.cat((padding_zero, dec_output), 1)

        dec_output_backward_zero = torch.cat((padding_zero, dec_output_backward), 1)

        # reverse dec_output_backward_zero

        dec_output_backward_zero_reverse = []

        for i in dec_output_backward_zero:
            dec_output_backward_zero_reverse.append(i.data.cpu().numpy()[::-1])

        dec_output_backward_zero_reverse = torch.tensor(np.array(dec_output_backward_zero_reverse))

        dec_output_backward_zero_reverse = tensorToCuda(dec_output_backward_zero_reverse)

        # dec_output_backward_zero_reverse=dec_output_backward_zero[:,::-1]
        # concat dec_output_zero  dec_output_backward_zero_reverse
        # [batch,tgt_len+1,2*d_model]

        dec_output_new = torch.cat((dec_output_zero, dec_output_backward_zero_reverse), 2)

        # LN
        dec_output_new = self.out_layer_norm(dec_output_new)

        # proj_share_weight=False
        return self.generator(dec_output_new, log_probs=log_probs)

    # ------------------------houq--------------------------------------------
    def extract_emb_feature(self, tgt_seq):
        forward_emb = self.decoder.embeddings.embeddings

        backward_emb = self.decoder_backward.embeddings.embeddings

        tgt_forward_emb = forward_emb(tgt_seq[:, :-1].contiguous())

        tgt_backward_emb = backward_emb(tgt_seq[:, 1:].contiguous())

        # padding zero
        batch_size, tgt_len, dim = tgt_forward_emb.size()

        padding_zero = torch.zeros(batch_size, 1, dim)

        padding_zero = tensorToCuda(padding_zero)

        # padding zero start
        tgt_forward_emb_zero = torch.cat((padding_zero, tgt_forward_emb), 1)

        # padding zero end
        tgt_backward_emb_zero = torch.cat((tgt_backward_emb, padding_zero), 1)

        # concat tgt_emb [batch,tgt_len+1,2*d_model]
        tgt_emb = torch.cat((tgt_forward_emb_zero, tgt_backward_emb_zero), 2)

        return tgt_emb

    # ------------------------houq--------------------------------------------
    def extract_hidden_feature(self, src_seq, tgt_seq):

        enc_output, enc_mask = self.encoder(src_seq)

        tgt_seq_forward = tgt_seq[:, :-1].contiguous()

        dec_output, _, _ = self.decoder(tgt_seq_forward, enc_output, enc_mask)

        # add backward self-attention

        tgt_seq_reverse = []

        for i in tgt_seq:
            tgt_seq_reverse.append(list(reversed(i.data.cpu().numpy())))

        tgt_seq_reverse = torch.tensor(np.array(tgt_seq_reverse))

        tgt_seq_reverse = tensorToCuda(tgt_seq_reverse)

        tgt_seq_backward = tgt_seq_reverse[:, :-1].contiguous()
        # tgt_seq_backward = tgt_seq[:, :0:-1].contiguous()

        dec_output_backward, _, _ = self.decoder_backward(tgt_seq_backward, enc_output, enc_mask)

        # padding zero

        batch_size, tgt_len, dim = dec_output.size()

        padding_zero = torch.zeros(batch_size, 1, dim)

        padding_zero = tensorToCuda(padding_zero)

        dec_output_zero = torch.cat((padding_zero, dec_output), 1)

        dec_output_backward_zero = torch.cat((padding_zero, dec_output_backward), 1)

        # reverse dec_output_backward_zero

        dec_output_backward_zero_reverse = []

        for i in dec_output_backward_zero:
            dec_output_backward_zero_reverse.append(i.data.cpu().numpy()[::-1])

        dec_output_backward_zero_reverse = torch.tensor(np.array(dec_output_backward_zero_reverse))

        dec_output_backward_zero_reverse = tensorToCuda(dec_output_backward_zero_reverse)

        # dec_output_backward_zero_reverse = dec_output_backward_zero[:, ::-1]

        # concat dec_output_zero  dec_output_backward_zero_reverse
        # [batch,tgt_len+1,2*d_model]

        dec_output_new = torch.cat((dec_output_zero, dec_output_backward_zero_reverse), 2)

        # LN
        # extract_feature : hidden state [batch,tgt_len,2*d_model]
        dec_output_new = self.out_layer_norm(dec_output_new)

        return dec_output_new

    # ------------------------houq--------------------------------------------
    def extract_logit_feature(self, src_seq, tgt_seq):

        dec_output = self.extract_hidden_feature(src_seq, tgt_seq)
        # extract_feature : logits vector [batch,tgt_len,n_words]
        logits = self.generator.cul_logits(dec_output)

        return logits

    def extract_fv(self, src_seq, tgt_seq):
        # extract_feature : hidden state [batch,tgt_len,2*d_model]
        dec_output = self.extract_hidden_feature(src_seq, tgt_seq)
        x_size = dec_output.size()
        # print(x_size)

        # proj_weight[n_vocab,2*d_model]
        proj_weight = self.generator.proj.linear.weight
        # print(proj_weight.size())

        # tgt_seq     [batch,tgt_len]
        tgt_seq = tgt_seq.view(-1)
        # tgt_seq     [batch*tgt_len]
        # print(tgt_seq.size())

        # word_proj_weight [batch*tgt_len,2d_model]
        word_proj_weight = proj_weight[tgt_seq]
        # print(word_proj_weight.size())

        word_proj_weight=word_proj_weight.view(x_size)

        fv = dec_output * word_proj_weight

        return fv

    # def forward(self, src_seq, tgt_seq, log_probs=True):
    #
    #     enc_output, enc_mask = self.encoder(src_seq)
    #     dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask)
    #
    #     return self.generator(dec_output, log_probs=log_probs)

    def encode(self, src_seq):

        ctx, ctx_mask = self.encoder(src_seq)

        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def init_decoder(self, enc_outputs, expand_size=1):

        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        if expand_size > 1:
            ctx = tile_batch(ctx, multiplier=expand_size)
            ctx_mask = tile_batch(ctx_mask, multiplier=expand_size)

        return {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "enc_attn_caches": None,
            "slf_attn_caches": None
        }

    def decode(self, tgt_seq, dec_states, log_probs=True):

        ctx = dec_states["ctx"]
        ctx_mask = dec_states['ctx_mask']
        enc_attn_caches = dec_states['enc_attn_caches']
        slf_attn_caches = dec_states['slf_attn_caches']

        dec_output, slf_attn_caches, enc_attn_caches = self.decoder(tgt_seq=tgt_seq, enc_output=ctx, enc_mask=ctx_mask,
                                                                    enc_attn_caches=enc_attn_caches,
                                                                    self_attn_caches=slf_attn_caches)

        next_scores = self.generator(dec_output[:, -1].contiguous(), log_probs=log_probs)

        dec_states['enc_attn_caches'] = enc_attn_caches
        dec_states['slf_attn_caches'] = slf_attn_caches

        return next_scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        slf_attn_caches = dec_states['slf_attn_caches']

        batch_size = slf_attn_caches[0][0].size(0) // beam_size

        n_head = self.decoder.n_head
        dim_per_head = self.decoder.dim_per_head

        slf_attn_caches = nest.map_structure(
            lambda t: tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=t,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, n_head, -1, dim_per_head]),
            slf_attn_caches)

        dec_states['slf_attn_caches'] = slf_attn_caches

        return dec_states
