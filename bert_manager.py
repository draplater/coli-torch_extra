import torch

from dataclasses import dataclass
from torch.nn import Module

from coli.basic_tools.common_utils import NullContextManager
from coli.basic_tools.dataclass_argparse import argfield
from coli.data_utils.dataset import SentenceFeaturesBase
from coli.torch_extra.utils import pad_and_stack_1d, broadcast_gather


class BERTPlugin(Module):
    @dataclass
    class Options(object):
        bert_model: str = argfield(predict_time=True)
        lower: bool = True

    def __init__(self, bert_model, lower=True, gpu=False, requires_grad=False):
        super().__init__()
        self.lower = lower
        self.requires_grad = requires_grad
        self.reload(bert_model, gpu)

    def reload(self, bert_model, gpu):
        from pytorch_pretrained_bert import BertTokenizer, BertModel
        if bert_model.endswith('.tar.gz'):
            self.tokenizer = BertTokenizer.from_pretrained(
                bert_model.replace('.tar.gz', '-vocab.txt'),
                do_lower_case=self.lower)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=self.lower)

        self.bert = BertModel.from_pretrained(bert_model)
        if gpu:
            self.bert = self.bert.cuda()
        self.output_dim = self.bert.pooler.dense.in_features
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings

    def process_sentence_feature(self, sent, sent_feature: SentenceFeaturesBase,
                                 padded_length):
        keep_boundaries = getattr(sent_feature, "bert_boundaries", False)
        word_pieces = ["[CLS]"]
        word_starts = []
        word_ends = []
        if keep_boundaries:
            word_starts.append(0)
            word_ends.append(0)

        for word in sent.words:
            pieces = self.tokenizer.tokenize(word)
            word_starts.append(len(word_pieces))
            word_pieces.extend(pieces)
            word_ends.append(len(word_pieces) - 1)

        word_pieces.append("[SEP]")
        if keep_boundaries:
            word_starts.append(len(word_pieces) - 1)
            word_ends.append(len(word_pieces) - 1)

        word_length = padded_length + (2 if keep_boundaries else 0)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_tokens"] = torch.tensor(self.tokenizer.convert_tokens_to_ids(word_pieces))
        sent_feature.extra["bert_word_starts"] = torch.zeros((word_length, ), dtype=torch.int64)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_word_starts"][:len(word_starts)] = torch.tensor(word_starts)
        sent_feature.extra["bert_word_ends"] = torch.zeros((word_length, ), dtype=torch.int64)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_word_ends"][:len(word_ends)] = torch.tensor(word_ends)

    def process_batch(self, pls, feed_dict, batch_sentences):
        feed_dict["bert_tokens"] = pad_and_stack_1d(
            [i.extra["bert_tokens"] for i in batch_sentences])
        feed_dict["bert_word_ends"] = torch.stack(
            [i.extra["bert_word_ends"] for i in batch_sentences])

    def forward(self, feed_dict):
        all_input_mask = feed_dict.bert_tokens.gt(0)
        with (torch.no_grad() if not self.requires_grad else NullContextManager()):
            all_encoder_layers, _ = self.bert(feed_dict.bert_tokens,
                                              attention_mask=all_input_mask)
            features = all_encoder_layers[-1]
            word_features = broadcast_gather(features, 1, feed_dict.bert_word_ends)
        return word_features
