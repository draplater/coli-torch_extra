import torch
from abc import ABCMeta, abstractproperty, abstractmethod
from pprint import pformat
from typing import List, Tuple, Generic, TypeVar, Type, Any, Optional

import numpy as np
from torch.nn import Module

from bilm.load_vocab import BiLMVocabLoader
from coli.basic_tools.common_utils import cache_result, T, NoPickle, AttrDict
from data_utils.embedding import ExternalEmbeddingLoader
from coli.basic_tools.logger import log_to_file
from parser_base import DependencyParserBase
from data_utils.dataset import SentenceFeaturesBase, bucket_types, SentenceBucketsBase

B = TypeVar("B", bound=SentenceBucketsBase)
U = TypeVar("U", bound=SentenceFeaturesBase)


class PyTorchParserBase(DependencyParserBase,
                        Generic[U],
                        metaclass=ABCMeta):
    sentence_feature_class: Any = abstractproperty()

    statistics: Any
    network: Module

    def __init__(self, args: Any, data_train):
        super(PyTorchParserBase, self).__init__(args, data_train)

        self.args = args
        self.hparams = args.hparams

        if self.args.bilm_path is not None:
            self.bilm_vocab = BiLMVocabLoader(self.args.bilm_path)
        else:
            self.bilm_vocab = None

        self.external_embedding_loader = NoPickle(
            ExternalEmbeddingLoader(args.embed_file)) \
            if args.embed_file is not None else None

        self.global_step = 0
        self.global_epoch = 0

    # noinspection PyMethodOverriding
    def train(self, train_bucket: B,
              dev_buckets: List[Tuple[str, "DataType", B]]):
        pass

    def sentence_convert_func(self, sent_idx: int, sentence: T,
                              padded_length: int):
        return self.sentence_feature_class.from_sentence_obj(
            sent_idx, sentence, self.statistics,
            self.external_embedding_loader.lookup
            if self.external_embedding_loader is not None else None,
            padded_length,
            bilm_loader=self.bilm_vocab
        )

    def create_bucket(self, bucket_type: Type[B], data: T, is_train) -> B:
        return bucket_type(
            data, self.sentence_convert_func,
            self.hparams.train_batch_size if is_train else self.hparams.test_batch_size,
            self.hparams.num_buckets, seed=self.hparams.seed,
            max_sentence_batch_size=self.hparams.max_sentence_batch_size
        )

    @abstractmethod
    def predict_bucket(self, test_buckets):
        raise NotImplementedError

    def predict(self, data_test):
        test_buckets = self.create_bucket(
            bucket_types[self.hparams.bucket_type], data_test, False)
        for i in self.predict_bucket(test_buckets):
            yield i

    @classmethod
    def repeat_train_and_validate(cls, data_train, data_devs, data_test, options):
        np.random.seed(options.hparams.seed)
        torch.manual_seed(options.hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.hparams.seed)
        parser = cls(options, data_train)
        log_to_file(parser.get_log_file(options))
        parser.logger.info('Options:\n%s', pformat(options.__dict__))
        parser.logger.info('Network:\n%s', pformat(parser.network))

        @cache_result(options.output + "/" + "data_cache_buckets.pkl",
                      enable=options.debug_cache)
        def load_data():
            train_bucket: B = parser.create_bucket(
                bucket_types[options.hparams.bucket_type],
                data_train, True)
            dev_buckets: List[B] = [
                parser.create_bucket(
                    bucket_types[options.hparams.bucket_type], data_dev, False)
                for data_dev in data_devs.values()]
            return train_bucket, dev_buckets

        train_buckets, dev_buckets = load_data()

        while True:
            current_step = parser.global_step
            if current_step > options.hparams.train_iters:
                break
            parser.train(
                train_buckets,
                [(a, b, c) for (a, b), c  # file_name, data, buckets
                 in zip(data_devs.items(), dev_buckets)]
            )

    def post_load(self, new_options):
        self.options.__dict__.update(new_options.__dict__)

    @classmethod
    def load(cls, prefix, new_options: Optional[AttrDict] = None):
        if new_options is None:
            new_options = AttrDict()

        self = torch.load(prefix, map_location="cpu" if not new_options.gpu else "cuda")
        self.post_load(new_options)
        return self

    def save(self, prefix, latest_filename=None):
        torch.save(self, prefix)
