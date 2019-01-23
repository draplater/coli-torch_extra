import torch
from torch.nn import Module, Embedding

from coli.data_utils import dataset
import torch.nn.functional as F

from coli.data_utils.dataset import SentenceFeaturesBase
from coli.data_utils.embedding import read_embedding


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
        [F.pad(i, pad_prefix + (to - i.size(0),), mode=mode, value=value)
         for i in inputs]
    )


class ExternalEmbeddingPlugin(Module):
    def __init__(self, embedding_filename, encoding="utf-8",
                 freeze=True, lower=False, gpu=False):
        super().__init__()
        self.lower = lower
        self.reload(embedding_filename, freeze, encoding, gpu)

    def reload(self, embedding_filename, freeze=True, encoding="utf-8", gpu=False):
        words_and_vectors = read_embedding(embedding_filename, encoding)
        self.dim = len(words_and_vectors[0][1])
        # noinspection PyCallingNonCallable
        words_and_vectors.insert(0, ("*UNK*", [0.0] * self.dim))

        words, vectors_py = zip(*words_and_vectors)
        self.lookup = {word: idx for idx, word in enumerate(words)}
        # noinspection PyCallingNonCallable
        vectors = torch.tensor(vectors_py, dtype=torch.float32)

        # noinspection PyReturnFromInit
        self.embedding = Embedding.from_pretrained(vectors, freeze=freeze)

        if gpu:
            self.embedding = self.embedding.cuda()

    def lower_func(self, word):
        if self.lower:
            return word.lower()
        return word

    def process_sentence_feature(self, sent, sent_feature,
                                 padded_length):
        sent_feature.extra["words_pretrained"] = lookup_list(
            (self.lower_func(i) for i in sent.words), self.lookup,
            padded_length=padded_length, default=0)

    def process_batch(self, pls, feed_dict, batch_sentences):
        feed_dict[pls.words_pretrained] = torch.stack([i.extra["words_pretrained"] for i in batch_sentences])

    def forward(self, feed_dict):
        return self.embedding(feed_dict.words_pretrained)
