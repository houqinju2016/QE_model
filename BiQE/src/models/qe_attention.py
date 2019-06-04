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
import torch
import torch.nn as nn
from src.modules.rnn import RNN
from src.data.vocabulary import PAD
import src.utils.init as my_init
from src.modules.sublayers import MultiHeadedAttention


class QE_ATTENTION(nn.Module):
    def __init__(self,
                 d_model, n_head,
                 feature_size=1024,
                 hidden_size=512,
                 dropout=0.0,
                 **kwargs
                 ):
        super(QE_ATTENTION, self).__init__()

        self.ctx_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=None)

        # Use PAD
        self.gru = RNN(type="gru", batch_first=True, input_size=feature_size, hidden_size=hidden_size,
                       bidirectional=True)
        self.lstm = RNN(type="lstm", batch_first=True, input_size=feature_size, hidden_size=hidden_size,
                        bidirectional=True)

        self.w = nn.Linear(2 * hidden_size, 1)
        my_init.default_init(self.w.weight)

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def cul_pre(self, context, mask):
        no_pad_mask = 1.0 - mask.float()

        # batch, seq_len, hidden_size_2 = context.size()
        # context_=context.view(batch, seq_len, 2, hidden_size_2 / 2)
        # context_forward = context_[:, :, 0]
        # context_backward = context_[:, :, 1]
        # context_forward_end = context_forward[:, -1]
        # context_backward_start = context_backward[:, 0]
        # ctx_mean = torch.cat((context_forward_end, context_backward_start), 1)

        ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)
        # ctx_mean = self.dropout(ctx_mean)
        pre = self.sigmoid(self.w(ctx_mean))
        # pre = self.w(ctx_mean)
        # pre = torch.clamp(pre, 0, 1)

        return pre

    def forward(self, emb, x, emb_bt, x_bt):
        batch_size, tgt_len = x.size()
        query_len = tgt_len

        src_len = emb_bt.size(1)
        enc_mask = x_bt.detach().eq(PAD)

        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)
        mid, attn, enc_attn_cache = self.ctx_attn(emb_bt, emb_bt, emb,
                                                  mask=dec_enc_attn_mask, enc_attn_cache=None)

        # mid_1=self.dropout(mid) + emb

        mid_1=torch.cat((self.dropout(mid),emb),2)

        x_mask = x.detach().eq(PAD)
        # dropout
        # mid = self.dropout(mid)

        ctx, _ = self.lstm(mid_1, x_mask)
        # dropout
        # ctx = self.dropout(ctx)

        pre = self.cul_pre(ctx, x_mask)

        return pre
