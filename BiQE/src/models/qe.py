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


class QE(nn.Module):
    def __init__(self,
                 feature_size=2052,
                 hidden_size=512,
                 dropout=0.0,
                 **kwargs
                 ):
        super(QE, self).__init__()

        # Use PAD
        self.gru = RNN(type="gru", batch_first=True, input_size=feature_size, hidden_size=hidden_size,
                       bidirectional=True)
        self.gru_bt = RNN(type="gru", batch_first=True, input_size=feature_size, hidden_size=hidden_size,
                       bidirectional=True)
        self.lstm = RNN(type="lstm", batch_first=True, input_size=feature_size, hidden_size=hidden_size,
                        bidirectional=True)
        self.lstm_bt = RNN(type="lstm", batch_first=True, input_size=feature_size, hidden_size=hidden_size,
                           bidirectional=True)
        self.w = nn.Linear(2 * hidden_size, 1)
        my_init.default_init(self.w.weight)

        self.w_all = nn.Linear(2 * 2 * hidden_size, 1)
        my_init.default_init(self.w_all.weight)

        self.w_1 = nn.Linear(2 * hidden_size, 1)
        my_init.default_init(self.w_1.weight)

        self.w_2 = nn.Linear(2 * hidden_size, 1)
        my_init.default_init(self.w_2.weight)

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

    def cul_pre_bt(self, context, mask, context_bt, mask_bt):
        no_pad_mask = 1.0 - mask.float()

        # batch, seq_len, hidden_size_2 = context.size()
        # context_=context.view(batch, seq_len, 2, hidden_size_2 / 2)
        # context_forward = context_[:, :, 0]
        # context_backward = context_[:, :, 1]
        # context_forward_end = context_forward[:, -1]
        # context_backward_start = context_backward[:, 0]
        # ctx_mean = torch.cat((context_forward_end, context_backward_start), 1)

        ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)

        no_pad_mask_bt = 1.0 - mask_bt.float()
        ctx_mean_bt = (context_bt * no_pad_mask_bt.unsqueeze(2)).sum(1) / no_pad_mask_bt.unsqueeze(2).sum(1)

        # ctx_mean = self.dropout(ctx_mean)

        # pre = self.sigmoid(self.w(ctx_mean))

        ctx_all = torch.cat((ctx_mean, ctx_mean_bt), 1)
        pre = self.sigmoid(self.w_all(ctx_all))
        # pre = self.sigmoid(self.w_1(ctx_mean)+self.w_2(ctx_mean_bt))
        # pre = self.w(ctx_mean)
        # pre = torch.clamp(pre, 0, 1)

        return pre

    def forward(self, emb, x, emb_bt, x_bt):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        x_mask = x.detach().eq(PAD)

        emb = self.dropout(emb)
        ctx, _ = self.lstm(emb, x_mask)
        ctx = self.dropout(ctx)

        x_mask_bt = x_bt.detach().eq(PAD)
        emb_bt = self.dropout(emb_bt)
        ctx_bt, _ = self.lstm_bt(emb_bt, x_mask_bt)
        # dropout
        ctx_bt = self.dropout(ctx_bt)

        pre = self.cul_pre_bt(ctx, x_mask, ctx_bt, x_mask_bt)

        return pre

    # def forward(self, emb, x):
    #     """
    #     :param x: Input sequence.
    #         with shape [batch_size, seq_len, input_size]
    #     """
    #     x_mask = x.detach().eq(PAD)
    #     # dropout
    #     emb = self.dropout(emb)
    #
    #     ctx, _ = self.lstm(emb, x_mask)
    #     # dropout
    #     ctx = self.dropout(ctx)
    #
    #     pre = self.cul_pre(ctx, x_mask)
    #
    #     return pre

