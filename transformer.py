import math
from typing import Optional

import torch
from dataclasses import dataclass
from torch import nn as nn
from torch.nn import init as init
import torch.nn.init as I

from coli.torch_span.layers import FeatureDropout, LayerNormalization


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.masked_fill_(attn_mask, -float('inf'))

        # Transposes to avoid https://github.com/pytorch/pytorch/issues/4893
        attn = self.softmax(attn.transpose(1, 2)).transpose(1, 2)
        # Note that this makes the distribution not sum to 1. At some point it
        # may be worth researching whether this is the right way to apply
        # dropout to the attention.
        # Note that the t2t code also applies dropout in this manner
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            self.w_qs1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_ks1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_vs1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_v // 2))

            self.w_qs2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_ks2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_vs2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_v // 2))

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            self.proj = nn.Linear(n_head * d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head * (d_v // 2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head * (d_v // 2), self.d_positional, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp):
        batch_size, max_sent_len, feature_count = inp.shape
        input_2d = inp.view(-1, feature_count)
        v_inp_repeated = input_2d.unsqueeze(0).repeat(self.n_head, 1, 1)
        qk_inp_repeated = v_inp_repeated

        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs)  # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks)  # n_head x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs)  # n_head x len_inp x d_v
        else:
            q_s = torch.cat([
                torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_qs1),
                torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_qs2),
            ], -1)
            k_s = torch.cat([
                torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_ks1),
                torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_ks2),
            ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:, :, :self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:, :, self.d_content:], self.w_vs2),
            ], -1)
        return q_s.view(self.n_head * batch_size, max_sent_len, -1), \
               k_s.view(self.n_head * batch_size, max_sent_len, -1), \
               v_s.view(self.n_head * batch_size, max_sent_len, -1)

    def combine_v(self, outputs):
        # Combine attention information from the different heads
        n_head = self.n_head

        if not self.partitioned:
            # Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

            # Project back to residual size
            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:, :, :d_v1]
            outputs2 = outputs[:, :, d_v1:]
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
            ], -1)

        return outputs

    def forward(self, inp, sentence_mask):
        batch_size, max_sent_len, feature_count = inp.shape
        residual = inp

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_padded, k_padded, v_padded = self.split_qkv_packed(inp)

        invalid_mask = ~sentence_mask
        attn_mask = invalid_mask.unsqueeze(1).expand(batch_size, max_sent_len, max_sent_len).repeat(self.n_head, 1, 1)

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask
        )

        outputs = self.combine_v(outputs_padded.view(self.n_head, batch_size * max_sent_len, self.d_v))

        outputs = outputs.view(batch_size, max_sent_len, -1)

        outputs = self.residual_dropout(outputs)

        return self.layer_norm(outputs + residual), attns_padded


class PositionwiseFeedForward(nn.Module):
    """
    A position-wise feed forward module.

    Projects to a higher-dimensional space before applying ReLU, then projects
    back.
    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)

        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output))
        output = self.w_2(output)

        output = self.residual_dropout(output)
        return self.layer_norm(output + residual)


class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff // 2)
        self.w_1p = nn.Linear(d_positional, d_ff // 2)
        self.w_2c = nn.Linear(d_ff // 2, self.d_content)
        self.w_2p = nn.Linear(d_ff // 2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        xc = x[:, :, :self.d_content]
        xp = x[:, :, self.d_content:]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc))
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp))
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output)
        return self.layer_norm(output + residual)


class TransformerEncoder(nn.Module):
    @dataclass
    class Options(object):
        num_layers: int = 8
        num_heads: int = 2
        d_kv: int = 32
        d_ff: int = 1024
        d_positional: Optional[int] = None
        num_layers_position_only: int = 0
        relu_dropout: float = 0.1
        residual_dropout: float = 0.1
        attention_dropout: float = 0.1
        timing_dropout: float = 0.0
        timing_method: str = "embedding"
        max_sent_len: int = 512

    def __init__(self, input_size,
                 num_layers=1, num_heads=2, d_kv=32, d_ff=1024,
                 d_positional=None,
                 num_layers_position_only=0,
                 relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1,
                 timing_dropout=0.0, timing_method="embedding",
                 max_sent_len=512):
        super().__init__()
        d_model = input_size
        d_k = d_v = d_kv
        self.output_dim = input_size

        self.stacks = []
        for i in range(num_layers):
            attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
                                      attention_dropout=attention_dropout, d_positional=d_positional)
            if d_positional is None:
                ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
                                             residual_dropout=residual_dropout)
            else:
                ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
                                                        residual_dropout=residual_dropout)

            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ff_{i}", ff)
            self.stacks.append((attn, ff))

        self.num_layers_position_only = num_layers_position_only
        if self.num_layers_position_only > 0:
            assert d_positional is None, "num_layers_position_only and partitioned are incompatible"

        self.timing_dropout = FeatureDropout(timing_dropout)

        if timing_method == "embedding":
            # Learned embeddings
            self.position_table = nn.Parameter(torch.FloatTensor(max_sent_len, input_size))
            I.normal_(self.position_table)
        else:
            assert timing_method == "sinusoidal"
            self.position_table = nn.Parameter(torch.zeros(max_sent_len, input_size), requires_grad=False)
            position = torch.arange(0, max_sent_len).view(-1).unsqueeze(-1).float()
            div_term = torch.exp((torch.arange(0, input_size, 2, dtype=torch.float) *
                             -(math.log(10000.0) / input_size)))
            self.position_table[:, 0::2] = torch.sin(position * div_term)
            self.position_table[:, 1::2] = torch.cos(position * div_term)

    def forward(self, res, lengths_or_mask, use_mask=False, timing_signal=None, add_timing_signal=True):
        if not use_mask:
            sentence_mask = sequence_mask(lengths_or_mask, res.shape[1])
        else:
            sentence_mask = lengths_or_mask
        if timing_signal is None:
            timing_signal = self.timing_dropout(
                self.position_table[:res.shape[1], :].unsqueeze(0).repeat(res.shape[0], 1, 1))

        if add_timing_signal:
            res += timing_signal

        for i, (attn, ff) in enumerate(self.stacks):
            if i >= self.num_layers_position_only:
                res, current_attns = attn(res, sentence_mask)
            else:
                res, current_attns = attn(res, qk_inp=timing_signal)
            res = ff(res)

        return res


Encoder = TransformerEncoder
