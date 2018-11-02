from typing import Type, Any

import numpy as np
import torch
from dataclasses import dataclass
from torch import Tensor
from torch.nn import Embedding, Module, Dropout, LayerNorm
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from coli.basic_tools.common_utils import combine_sub_options
from torch_extra.seq_utils import sort_sequences, unsort_sequences, pad_timestamps_and_batches
from torch_extra import tf_rnn
from coli.basic_tools.logger import default_logger


def get_external_embedding(loader, freeze=True):
    vectors_np = np.array(loader.vectors)
    vectors_np = vectors_np / np.std(loader.vectors)
    return Embedding.from_pretrained(
        torch.FloatTensor(vectors_np), freeze=freeze)


@dataclass
class BasicCharEmbeddingOptions(object):
    dim_char: "Character dims" = 100
    max_char: "max characters" = 20


class ContextualUnits(object):
    @dataclass
    class Options(object):
        contextual_unit: "contextual unit" = "lstm"
        lstm_size: "LSTM dimension" = 500
        layer_norm: "use layer normalization" = False
        lstm_layers: "lstm layer count" = 2
        input_keep_prob: "input keep prob" = 0.5
        recurrent_keep_prob: "recurrent keep prob" = 0.5

    @classmethod
    def get(cls, input_size, hparams):
        return contextual_units[hparams.contextual_unit](
            input_size=input_size,
            hidden_size=hparams.lstm_size,
            num_layers=hparams.lstm_layers,
            input_keep_prob=hparams.input_keep_prob,
            recurrent_keep_prob=hparams.recurrent_keep_prob,
            layer_norm=hparams.layer_norm)


class LSTMLayer(Module):
    default_cell = torch.nn.LSTM

    @dataclass
    class Options(ContextualUnits.Options):
        """ LSTM Layer Options"""
        pass

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


class CharLSTMLayer(Module):
    @dataclass
    class Options(BasicCharEmbeddingOptions):
        char_lstm_layers: "Character LSTM layer count" = 2

    def __init__(self, options: Options):
        super(CharLSTMLayer, self).__init__()
        self.options = options
        self.char_lstm = LSTMLayer(input_size=options.dim_char,
                                   hidden_size=options.dim_char // 2,
                                   num_layers=options.char_lstm_layers,
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
            batch_size, max_sent_length, self.options.dim_char)


class CharacterEmbedding(object):
    classes = {"rnn": CharLSTMLayer}
    default = "rnn"

    @dataclass
    class Options(combine_sub_options(classes)):
        char_embedding_type: "Character Embedding Type" = "rnn"

    @staticmethod
    def get_char_embedding(hparams: Options):
        if hparams.char_embedding_type == "rnn":
            return CharLSTMLayer(hparams)
        else:
            assert hparams.char_embedding_type is None, \
                "invalid char_embeded %s" % hparams.char_embedding_type


class TimeDistributed(Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_tensor):
        batch_size, time_steps, features = input_tensor.shape
        output = self.inner(
            input_tensor.view(batch_size * time_steps, features)).view(
            batch_size, time_steps, -1
        )
        return output


contextual_units = {"lstm": LSTMLayer, "gru": GRULayer,
                    "tf-lstm": tf_rnn.LSTM}
ContextualUnitsOptions: Type[Any] = combine_sub_options(
    contextual_units, name="ContextualUnitsOptions")


def cross_encropy(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits, labels, reduction='none')


loss_funcs = {"softmax": cross_encropy}
