from typing import Optional

import torch

from dataclasses import dataclass
from torch.nn import Linear

from coli.basic_tools.common_utils import NullContextManager
from coli.basic_tools.dataclass_argparse import argfield, OptionsBase
from coli.data_utils.dataset import SentenceFeaturesBase, START_OF_SENTENCE, END_OF_SENTENCE
from coli.torch_extra.dataset import InputPluginBase
from coli.torch_extra.utils import pad_and_stack_1d, broadcast_gather
from coli.torch_span.layers import FeatureDropout

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
}


class BERTPlugin(InputPluginBase):
    @dataclass
    class Options(OptionsBase):
        bert_model: str = argfield(predict_time=True)
        lower: bool = True
        project_to: Optional[int] = None
        feature_dropout: float = 0.0
        pooling_method: Optional[str] = "last"

    def __init__(self, bert_model, lower=True, project_to=None,
                 feature_dropout=0.0,
                 pooling_method="last",
                 gpu=False, requires_grad=False):
        super().__init__()
        self.lower = lower
        self.requires_grad = requires_grad
        self.pooling_method = pooling_method

        self.reload(bert_model, gpu)

        if feature_dropout > 0:
            self.feature_dropout_layer = FeatureDropout(feature_dropout)

        if project_to:
            self.projection = Linear(self.output_dim, project_to, bias=False)
            self.output_dim = project_to
        else:
            self.projection = None

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
                                 padded_length, start_and_end=False):
        keep_boundaries = start_and_end
        pooling_method = getattr(self, "pooling_method", "last")
        word_pieces = ["[CLS]"]
        word_starts = []
        word_ends = []
        if keep_boundaries:
            word_starts.append(0)
            word_ends.append(0)

        for maybe_words in sent.words:
            word_starts.append(len(word_pieces))
            if maybe_words == START_OF_SENTENCE or maybe_words == END_OF_SENTENCE:
                continue
            for word in maybe_words.split("_"):  # like more_than in deepbank:
                word = BERT_TOKEN_MAPPING.get(word, word)
                pieces = self.tokenizer.tokenize(word)
                word_pieces.extend(pieces)
            word_ends.append(len(word_pieces) - 1)

        word_pieces.append("[SEP]")
        if keep_boundaries:
            word_starts.append(len(word_pieces) - 1)
            word_ends.append(len(word_pieces) - 1)

        word_length = padded_length + (2 if keep_boundaries else 0)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_tokens"] = torch.tensor(self.tokenizer.convert_tokens_to_ids(word_pieces))
        sent_feature.extra["bert_word_starts"] = torch.zeros((word_length,), dtype=torch.int64)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_word_starts"][:len(word_starts)] = torch.tensor(word_starts)
        sent_feature.extra["bert_word_ends"] = torch.zeros((word_length,), dtype=torch.int64)
        # noinspection PyCallingNonCallable
        sent_feature.extra["bert_word_ends"][:len(word_ends)] = torch.tensor(word_ends)

    def process_batch(self, pls, feed_dict, batch_sentences):
        pooling_method = getattr(self, "pooling_method", "last")
        feed_dict["bert_tokens"] = pad_and_stack_1d(
            [i.extra["bert_tokens"] for i in batch_sentences])
        feed_dict["bert_word_ends"] = torch.stack(
            [i.extra["bert_word_ends"] for i in batch_sentences])

    def forward(self, feed_dict):
        pooling_method = getattr(self, "pooling_method", "last")
        all_input_mask = feed_dict.bert_tokens.gt(0)
        with (torch.no_grad() if not self.requires_grad else NullContextManager()):
            all_encoder_layers, _ = self.bert(feed_dict.bert_tokens,
                                              attention_mask=all_input_mask)
            features = all_encoder_layers[-1]
            if pooling_method == "last":
                word_features = broadcast_gather(features, 1, feed_dict.bert_word_ends)
            elif pooling_method == "none":
                word_features = features
            else:
                raise Exception(f"Invalid pooling method {pooling_method}")

        if self.projection is not None:
            word_features = self.projection(word_features)

        if hasattr(self, "feature_dropout_layer"):
            word_features = self.feature_dropout_layer(word_features)

        # noinspection PyUnboundLocalVariable
        return word_features
