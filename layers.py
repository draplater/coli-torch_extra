import numpy as np
import torch
from dataclasses import dataclass, field
from torch import Tensor
from torch.nn import Embedding, Module, Dropout, LayerNorm, Sequential, Linear, ReLU
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from coli.basic_tools.dataclass_argparse import argfield, BranchSelect
from coli.torch_extra.seq_utils import sort_sequences, unsort_sequences, pad_timestamps_and_batches
from coli.torch_extra import tf_rnn
from coli.basic_tools.logger import default_logger
from coli.torch_extra.transformer import TransformerEncoder


def get_external_embedding(loader, freeze=True):
    vectors_np = np.array(loader.vectors)
    vectors_np = vectors_np / np.std(loader.vectors)
    return Embedding.from_pretrained(
        torch.FloatTensor(vectors_np), freeze=freeze)


class LSTMLayer(Module):
    default_cell = torch.nn.LSTM

    @dataclass
    class Options(object):
        """ LSTM Layer Options"""
        hidden_size: "LSTM dimension" = 500
        num_layers: "lstm layer count" = 2
        input_keep_prob: "input keep prob" = 0.5
        recurrent_keep_prob: "recurrent keep prob" = 0.5
        layer_norm: "use layer normalization" = False

    def __init__(self, input_size, hidden_size, num_layers,
                 input_keep_prob,
                 recurrent_keep_prob,
                 layer_norm=False
                 ):
        super(LSTMLayer, self).__init__()
        if recurrent_keep_prob != 1.0:
            raise NotImplementedError(
                "Pytorch RNN does not support recurrent dropout.")
        if layer_norm:
            default_logger.warning("Pytorch RNN only support layer norm at last layer.")

        self.rnn = self.default_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=1 - input_keep_prob,
            bidirectional=True)

        self.layer_norm = LayerNorm(hidden_size * 2) if layer_norm else None
        self.first_dropout = Dropout(1 - input_keep_prob)
        self.reset_parameters()
        self.output_dim = hidden_size * 2

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                # set bias to 0
                param.data.fill_(0)
                # set forget bias to 1
                # n = param.size(0)
                # start, end = n // 4, n // 2
                # param.data[start:end].fill_(1.)
            # elif "weight_hh" in name:
            #     for i in range(0, param.size(0), self.rnn.hidden_size):
            #         torch.nn.init.orthogonal_(param[i:i+self.rnn.hidden_size].data,gain=1)
            else:
                # torch.nn.init.xavier_normal_(param.data)
                torch.nn.init.orthogonal_(param.data)

    def forward(self, seqs: Tensor, lengths, return_sequence=True, is_sorted=False):
        seqs = self.first_dropout(seqs)
        if not is_sorted:
            packed_seqs, unsort_idx = sort_sequences(seqs, lengths)
        else:
            packed_seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
            unsort_idx = None
        output_pack, state_n = self.rnn(packed_seqs)

        if return_sequence:
            # return seqs
            if not is_sorted:
                ret = unsort_sequences(output_pack, unsort_idx, seqs.shape)
            else:
                output_seqs, _ = pad_packed_sequence(output_pack, batch_first=True)
                ret = pad_timestamps_and_batches(output_seqs, seqs.shape)
            if self.layer_norm is not None:
                ret = self.layer_norm(ret)
            return ret
        else:
            # return final states
            # ignore layer norm
            if isinstance(state_n, tuple) and len(state_n) == 2:
                # LSTM
                h_n, c_n = state_n
            else:
                # GRU
                h_n = state_n
            _, word_count, hidden_size = h_n.shape
            ret = h_n[-2:].transpose(0, 1).contiguous().view(word_count, hidden_size * 2)
            if seqs.shape[0] is not None and word_count < seqs.shape[0]:
                ret = F.pad(ret,
                            (0, 0,
                             0, seqs.shape[0] - word_count
                             ))
            if not is_sorted:
                return ret.index_select(0, unsort_idx)
            else:
                return ret


class GRULayer(LSTMLayer):
    default_cell = torch.nn.GRU


contextual_units = {"lstm": LSTMLayer, "gru": GRULayer,
                    "tf-lstm": tf_rnn.LSTM, "transformer": TransformerEncoder}


class ContextualUnits(BranchSelect):
    branches = contextual_units

    @dataclass
    class Options(BranchSelect.Options):
        type: "contextual unit" = argfield("lstm", choices=contextual_units)
        lstm_options: LSTMLayer.Options = field(default_factory=LSTMLayer.Options)
        gru_options: GRULayer.Options = field(default_factory=LSTMLayer.Options)
        transformer_options: TransformerEncoder.Options = field(default_factory=TransformerEncoder.Options)


class CharLSTMLayer(Module):
    @dataclass
    class Options(object):
        num_layers: "Character LSTM layer count" = 2

    def __init__(self, input_size, num_layers):
        super(CharLSTMLayer, self).__init__()
        self.char_lstm = LSTMLayer(input_size=input_size,
                                   hidden_size=input_size // 2,
                                   num_layers=num_layers,
                                   input_keep_prob=1.0,
                                   recurrent_keep_prob=1.0
                                   )

    def forward(self, char_lengths, char_embeded_4d: Tensor, reuse=False):
        batch_size, max_sent_length, max_characters, embed_size = char_embeded_4d.shape
        char_lengths_1d = char_lengths.view(batch_size * max_sent_length)

        # batch_size * bucket_size, max_characters, embed_size
        char_embeded_3d = char_embeded_4d.view(
            batch_size * max_sent_length,
            max_characters, embed_size)
        return self.char_lstm(char_embeded_3d, char_lengths_1d,
                              return_sequence=False).view(
            batch_size, max_sent_length, -1)


char_embeddings = {"rnn": CharLSTMLayer}


class CharacterEmbedding(BranchSelect):
    branches = char_embeddings

    @dataclass
    class Options(object):
        type: "Character Embedding Type" = argfield("rnn", choices=char_embeddings)
        rnn_options: CharLSTMLayer.Options = field(default_factory=CharLSTMLayer.Options)
        dim_char: int = 100
        max_char: int = 20


def cross_encropy(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits, labels, reduction='none')


loss_funcs = {"softmax": cross_encropy}


@dataclass
class AdvancedLearningOptions(object):
    learning_rate: float = 1e-3
    learning_rate_warmup_steps: int = 160
    step_decay_factor: float = 0.5
    step_decay_patience: int = 5

    clip_grad_norm: float = 0.0


def create_mlp(input_dim, output_dim, hidden_dims=(), activation=ReLU,
               layer_norm=False,
               last_bias=True):
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    module_list = []
    for i in range(len(dims) - 1):
        if i == len(dims) - 2:
            use_bias = last_bias
        else:
            use_bias = True
        linear = Linear(dims[i], dims[i + 1], use_bias)
        torch.nn.init.xavier_normal_(linear.weight)
        if use_bias:
            torch.nn.init.zeros_(linear.bias)
        module_list.append(linear)
        if i != len(dims) - 2:
            if layer_norm:
                module_list.append(LayerNorm(dims[i + 1]))
            module_list.append(activation())
    return Sequential(*module_list)
