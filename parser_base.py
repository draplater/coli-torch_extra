import torch
from abc import ABCMeta, abstractproperty
from pprint import pformat
from typing import List, Tuple, Generic, TypeVar, Type, Any

import numpy as np

from bilm.load_vocab import BiLMVocabLoader
from logger import log_to_file
from parser_base import DependencyParserBase
from data_utils.dataset import SentenceBuckets

U = TypeVar("U", bound=SentenceBuckets)


class PyTorchParserBase(DependencyParserBase,
                        Generic[U],
                        metaclass=ABCMeta):
    bucket_class: Type[U] = abstractproperty()

    statistics: Any
    bilm_vocab: BiLMVocabLoader
    external_embedding_loader: Any
    global_step: int

    def train(self, train_bucket: SentenceBuckets,
              dev_buckets: List[Tuple[str, "DataType", U]]):
        pass

    @classmethod
    def repeat_train_and_validate(cls, data_train, data_devs, data_test, options):
        np.random.seed(options.hparams.seed)
        torch.manual_seed(options.hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.hparams.seed)
        parser = cls(options, data_train)
        log_to_file(parser.get_log_file(options))
        parser.logger.info('Options:\n%s', pformat(options.__dict__))
        train_buckets: SentenceBuckets = cls.bucket_class(
            data_train, parser.statistics, parser.external_embedding_loader.lookup,
            options.hparams.train_batch_size,
            options.hparams.num_buckets, seed=options.hparams.seed,
            bilm_loader=parser.bilm_vocab,
            max_sentence_batch_size=options.hparams.max_sentence_batch_size
        )
        dev_buckets: List[SentenceBuckets] = [cls.bucket_class(
            data_dev, parser.statistics, parser.external_embedding_loader.lookup,
            options.hparams.test_batch_size,
            options.hparams.num_valid_bkts, seed=options.hparams.seed,
            bilm_loader=parser.bilm_vocab,
            max_sentence_batch_size=options.hparams.max_sentence_batch_size
        )
            for data_dev in data_devs.values()]
        while True:
            current_step = parser.global_step
            if current_step > options.hparams.train_iters:
                break
            parser.train(
                train_buckets,
                [(a, b, c) for (a, b), c  # file_name, data, buckets
                 in zip(data_devs.items(), dev_buckets)]
            )
