import torch

from coli.data_utils import dataset
import torch.nn.functional as F


def lookup_list(tokens_itr, token_dict, padded_length,
                default, dtype=torch.int64,
                start_and_stop=False,
                tensor_factory=torch.zeros):
    return dataset.lookup_list(tokens_itr, token_dict, padded_length,
                               default, dtype, start_and_stop, tensor_factory)


def lookup_characters(words_itr, char_dict, padded_length,
                      default, max_word_length=20, dtype=torch.int64,
                      start_and_stop=True,
                      sentence_start_and_stop=False,
                      return_lengths=False,
                      tensor_factory=torch.zeros
                      ):
    return dataset.lookup_characters(words_itr, char_dict, padded_length,
                                     default, max_word_length, dtype,
                                     start_and_stop, sentence_start_and_stop,
                                     return_lengths, tensor_factory)


def pad_and_stack(inputs, to=None, mode="constant", value=0):
    pad_prefix = tuple(0 for _ in range(2 * len(inputs[0].shape) - 1))
    return torch.stack(
        [F.pad(i, pad_prefix + (to - i.size(0), ), mode=mode, value=value)
         for i in inputs]
    )
