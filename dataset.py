import torch
from torch.nn import Module, Embedding, Linear

from coli.data_utils import dataset
import torch.nn.functional as F

from coli.data_utils.dataset import SentenceFeaturesBase
from coli.data_utils.embedding import read_embedding
from coli.torch_extra.utils import pad_and_stack_1d


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


class ExternalEmbeddingPlugin(Module):
    def __init__(self, embedding_filename, project_to=None, encoding="utf-8",
                 freeze=True, lower=False, gpu=False):
        super().__init__()
        self.lower = lower
        self.project_to = project_to
        self.reload(embedding_filename, freeze, encoding, gpu)

    def reload(self, embedding_filename, freeze=True, encoding="utf-8", gpu=False):
        words_and_vectors = read_embedding(embedding_filename, encoding)
        self.output_dim = len(words_and_vectors[0][1])
        # noinspection PyCallingNonCallable
        words_and_vectors.insert(0, ("*UNK*", [0.0] * self.output_dim))

        words, vectors_py = zip(*words_and_vectors)
        self.lookup = {word: idx for idx, word in enumerate(words)}
        # noinspection PyCallingNonCallable
        vectors = torch.tensor(vectors_py, dtype=torch.float32)

        # noinspection PyReturnFromInit
        self.embedding = Embedding.from_pretrained(vectors, freeze=freeze)

        if gpu:
            self.embedding = self.embedding.cuda()

        if self.project_to:
            self.projection = Linear(self.output_dim, self.project_to)
            self.output_dim = self.project_to

    def lower_func(self, word):
        if self.lower:
            return word.lower()
        return word

    def process_sentence_feature(self, sent, sent_feature,
                                 padded_length, start_and_stop=False):
        words = [self.lower_func(i) for i in sent.words]
        sent_feature.extra["words_pretrained"] = lookup_list(
            words, self.lookup,
            padded_length=padded_length, default=0, start_and_stop=start_and_stop)

    def process_batch(self, pls, feed_dict, batch_sentences):
        feed_dict[pls.words_pretrained] = pad_and_stack_1d([i.extra["words_pretrained"] for i in batch_sentences])

    def forward(self, feed_dict):
        ret = self.embedding(feed_dict.words_pretrained)
        if hasattr(self, "projection"):
            ret = self.projection(ret)
        return ret
