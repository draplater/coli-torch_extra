import argparse
import pickle
import random
from typing import Generic, TypeVar, Type, Dict, List

import six
from io import open

from argparse import ArgumentParser
from pprint import pformat

import os
import sys
import subprocess
from abc import ABCMeta, abstractmethod, abstractproperty

import time

import graph_utils
import tree_utils
from common_utils import set_proc_name, ensure_dir, smart_open, NoPickle
from logger import get_logger, default_logger, log_to_file
from training_scheduler import TrainingScheduler


class DataTypeBase(metaclass=ABCMeta):
    @abstractmethod
    def from_file(self, file_path: str, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def to_string(self):
        raise NotImplementedError


U = TypeVar("U", bound=DataTypeBase)


class DependencyParserBase(Generic[U], metaclass=ABCMeta):
    DataType: Type[U] = None
    available_data_formats: Dict[str, Type[U]] = {}
    default_data_format_name = "default"

    def __init__(self, options, data_train=None, *args, **kwargs):
        super(DependencyParserBase, self).__init__()
        self.options = options
        # do not log to console if not training
        self.log_to_file = NoPickle(data_train is not None)

    @property
    def logger(self):
        if getattr(self, "_logger", None) is None:
            self._logger = NoPickle(
                self.get_logger(self.options,
                                log_to_file=self.log_to_file))
        return self._logger

    @classmethod
    def get_data_formats(cls) -> Dict[str, Type[U]]:
        """ for old class which has "DataType" but not "available_data_formats" """
        if not cls.available_data_formats:
            return {"default": cls.DataType}
        else:
            return cls.available_data_formats

    @abstractmethod
    def train(self, graphs: List[U], *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, graphs):
        """:rtype: list[self.DataType]"""
        pass

    @abstractmethod
    def save(self, prefix):
        pass

    @classmethod
    @abstractmethod
    def load(cls, prefix, new_options=None):
        pass

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        group = arg_parser.add_argument_group(DependencyParserBase.__name__)
        group.add_argument("--title", type=str, dest="title", default="default")
        group.add_argument("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                           required=True)
        group.add_argument("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", nargs="+",
                           required=True)
        group.add_argument("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE")
        group.add_argument("--outdir", type=str, dest="output", required=True)
        group.add_argument("--max-save", type=int, dest="max_save", default=100)
        group.add_argument("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model.")
        group.add_argument("--epochs", type=int, dest="epochs", default=30)
        group.add_argument("--lr", type=float, dest="learning_rate", default=None)
        group.add_argument("--print-every", type=int, default=100)

    @classmethod
    def add_predict_arguments(cls, arg_parser):
        group = arg_parser.add_argument_group(DependencyParserBase.__name__)
        group.add_argument("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", required=True)
        group.add_argument("--output", dest="out_file", help="Output file", metavar="FILE", required=True)
        group.add_argument("--model", dest="model", help="Load/Save model file", metavar="FILE", required=True)
        group.add_argument("--eval", action="store_true", dest="evaluate", default=False)
        group.add_argument("--format", dest="input_format", choices=["standard", "tokenlist",
                                                                     "space", "english", "english-line"],
                           help='Input format. (default)"standard": use the same format of treebank;\n'
                                'tokenlist: like [[(sent_1_word1, sent_1_pos1), ...], [...]];\n'
                                'space: sentence is separated by newlines, and words are separated by space;'
                                'no POSTag info will be used. \n'
                                'english: raw english sentence that will be processed by NLTK tokenizer, '
                                'no POSTag info will be used.',
                           default="standard"
                           )

    @classmethod
    def add_common_arguments(cls, arg_parser):
        group = arg_parser.add_argument_group(DependencyParserBase.__name__ + "(train and test)")
        group.add_argument("--dynet-seed", type=int, dest="seed", default=0)
        group.add_argument("--dynet-autobatch", type=int, dest="autobatch", default=0)
        group.add_argument("--dynet-mem", dest="mem", default=0)
        group.add_argument("--dynet-gpus", type=int, dest="mem", default=0)
        group.add_argument("--dynet-l2", type=float, dest="l2", default=0.0)
        group.add_argument("--dynet-weight-decay", type=float, dest="weight_decay", default=0.0)
        group.add_argument("--output-scores", action="store_true", dest="output_scores", default=False)
        group.add_argument("--data-format", dest="data_format",
                           choices=cls.get_data_formats(),
                           default=cls.default_data_format_name)
        group.add_argument("--bilm-cache")
        group.add_argument("--bilm-use-cache-only", action="store_true", default=False)
        group.add_argument("--bilm-path", metavar="FILE")
        group.add_argument("--bilm-stateless", action="store_true", default=False)
        group.add_argument("--bilm-gpu", default="")

    @classmethod
    def options_hook(cls, options):
        pass

    def get_log_file(self, options):
        if getattr(self, "logger_timestamp", None) is None:
            self.logger_timestamp = int(time.time())
        return os.path.join(
            options.output,
            "{}_{}_train.log".format(options.title, self.logger_timestamp))

    def get_logger(self, options, log_to_console=True, log_to_file=True):
        return get_logger(files=self.get_log_file(options) if log_to_file else None,
                          log_to_console=log_to_console,
                          name=getattr(options, "title", "logger"))

    @classmethod
    def train_parser(cls, options, data_train=None, data_dev=None, data_test=None):
        if sys.platform.startswith("linux"):
            set_proc_name(options.title)
        default_logger.name = options.title
        ensure_dir(options.output)

        cls.options_hook(options)
        DataFormatClass = cls.get_data_formats()[options.data_format]

        if data_train is None:
            data_train = DataFormatClass.from_file(options.conll_train)

        if data_dev is None:
            data_dev = {i: DataFormatClass.from_file(i, False) for i in options.conll_dev}

        if data_test is None and options.conll_test is not None:
            data_test = DataFormatClass.from_file(options.conll_test, False)
        else:
            data_test = None

        if options.bilm_cache is not None:
            if not os.path.exists(options.bilm_cache):
                train_sents = set(tuple(sent.words) for sent in data_train)
                dev_sentences = set()
                for one_data_dev in data_dev.values():
                    dev_sentences.update(set(tuple(sent.words) for sent in one_data_dev))
                if data_test is not None:
                    dev_sentences.update(set(tuple(sent.words) for sent in data_test))
                dev_sentences -= train_sents
                default_logger.info("Considering {} training sentences and {} dev sentences for bilm cache".format(
                    len(train_sents), len(dev_sentences)))
                # avoid running tensorflow in current process
                script_path = os.path.join(os.path.dirname(__file__), "bilm/cache_manager.py")
                p = subprocess.Popen([sys.executable, script_path, "pickle"], stdin=subprocess.PIPE, stdout=sys.stdout,
                                     stderr=sys.stderr)
                args = (options.bilm_path, options.bilm_cache, train_sents, dev_sentences, options.bilm_gpu)
                p.communicate(pickle.dumps(args))
                # pickle.dump(args, p.stdin)
                if p.returncode != 0:
                    raise Exception("Error when generating bilm cache.")
        try:
            os.makedirs(options.output)
        except OSError:
            pass

        return cls.repeat_train_and_validate(
            data_train, data_dev, data_test, options)

    @classmethod
    def repeat_train_and_validate(cls, data_train, data_devs, data_test, options):
        DataFormatClass = cls.get_data_formats()[options.data_format]
        # noinspection PyArgumentList
        parser = cls(options, data_train)
        log_to_file(parser.get_log_file(options))
        parser.logger.info('Options:\n%s', pformat(options.__dict__))
        random_obj = random.Random(1)
        for epoch in range(options.epochs):
            parser.logger.info('Starting epoch %d', epoch)
            random_obj.shuffle(data_train)
            options.is_train = True
            parser.train(data_train)

            # save model and delete old model
            for i in range(0, epoch - options.max_save):
                path = os.path.join(options.output, os.path.basename(options.model)) + str(i + 1)
                if os.path.exists(path):
                    os.remove(path)
            path = os.path.join(options.output, os.path.basename(options.model)) + str(epoch + 1)
            parser.save(path)

            def predict(sentences, gold_file, output_file):
                options.is_train = False
                with open(output_file, "w") as f_output:
                    if hasattr(DataFormatClass, "file_header"):
                        f_output.write(DataFormatClass.file_header + "\n")
                    for i in parser.predict(sentences):
                        f_output.write(i.to_string())
                # script_path = os.path.join(os.path.dirname(__file__), "main.py")
                # p = subprocess.Popen([sys.executable, script_path, "mst+empty", "predict", "--model", path,
                #                       "--test", gold_file,
                #                       "--output", output_file], stdout=sys.stdout)
                # p.wait()
                DataFormatClass.evaluate_with_external_program(gold_file, output_file)

            for file_name, file_content in data_devs.items():
                dev_output = cls.get_output_name(options.output, file_name, epoch)
                predict(file_content, file_name, dev_output)

    @classmethod
    def get_output_name(cls, out_dir, file_name, epoch):
        try:
            prefix, suffix = os.path.basename(file_name).rsplit(".", 1)
        except ValueError:
            prefix = os.path.basename(file_name)
            suffix = ""
        dev_output = os.path.join(
            out_dir,
            '{}_epoch_{}.{}'.format(prefix, epoch + 1, suffix))
        return dev_output

    @classmethod
    def predict_with_parser(cls, options):
        DataFormatClass = cls.get_data_formats()[options.data_format]
        if options.input_format == "standard":
            data_test = DataFormatClass.from_file(options.conll_test, False)
        elif options.input_format == "space":
            with smart_open(options.conll_test) as f:
                data_test = [DataFormatClass.from_words_and_postags([(word, "X") for word in line.strip().split(" ")])
                             for line in f]
        elif options.input_format.startswith("english"):
            from nltk import download, sent_tokenize
            from nltk.tokenize import TreebankWordTokenizer
            download("punkt")
            with smart_open(options.conll_test) as f:
                raw_sents = []
                for line in f:
                    if options.input_format == "english-line":
                        raw_sents.append(line.strip())
                    else:
                        this_line_sents = sent_tokenize(line.strip())
                        raw_sents.extend(this_line_sents)
                tokenized_sents = TreebankWordTokenizer().tokenize_sents(raw_sents)
                data_test = [DataFormatClass.from_words_and_postags([(token, "X") for token in sent])
                             for sent in tokenized_sents]
        elif options.input_format == "tokenlist":
            with smart_open(options.conll_test) as f:
                items = eval(f.read())
            data_test = DataFormatClass.from_words_and_postags(items)
        else:
            raise ValueError("invalid format option")

        default_logger.info('Loading Model...')
        options.is_train = False
        parser = cls.load(options.model, options)
        parser.logger.info('Model loaded')

        ts = time.time()
        with smart_open(options.out_file, "w") as f_output:
            if hasattr(DataFormatClass, "file_header"):
                f_output.write(DataFormatClass.file_header + "\n")
            for i in parser.predict(data_test):
                f_output.write(i.to_string())
        te = time.time()
        parser.logger.info('Finished predicting and writing test. %.2f seconds.', te - ts)

        if options.evaluate:
            DataFormatClass.evaluate_with_external_program(options.conll_test,
                                                           options.out_file)

    @classmethod
    def get_arg_parser(cls):
        parser = ArgumentParser(sys.argv[0])
        cls.fill_arg_parser(parser)
        return parser

    @classmethod
    def fill_arg_parser(cls, parser):
        sub_parsers = parser.add_subparsers()
        sub_parsers.required = True
        sub_parsers.dest = 'mode'

        # Train
        train_subparser = sub_parsers.add_parser("train")
        cls.add_parser_arguments(train_subparser)
        cls.add_common_arguments(train_subparser)
        train_subparser.set_defaults(func=cls.train_parser)

        # Predict
        predict_subparser = sub_parsers.add_parser("predict")
        cls.add_predict_arguments(predict_subparser)
        cls.add_common_arguments(predict_subparser)
        predict_subparser.set_defaults(func=cls.predict_with_parser)

        eval_subparser = sub_parsers.add_parser("eval")
        eval_subparser.add_argument("--data-format", dest="data_format",
                                    choices=cls.get_data_formats(),
                                    default=cls.default_data_format_name)
        eval_subparser.add_argument("gold")
        eval_subparser.add_argument("system")
        eval_subparser.set_defaults(func=cls.eval_only)

    @classmethod
    def get_training_scheduler(cls, train=None, dev=None, test=None):
        return TrainingScheduler(cls.train_parser, cls, train, dev, test)

    @classmethod
    def eval_only(cls, options):
        DataFormatClass = cls.get_data_formats()[options.data_format]
        DataFormatClass.evaluate_with_external_program(options.gold, options.system)

    @classmethod
    def get_next_arg_parser(cls, stage, options):
        return None

    @classmethod
    def fill_missing_params(cls, options):
        test_arg_parser = ArgumentParser()
        cls.add_parser_arguments(test_arg_parser)
        cls.add_common_arguments(test_arg_parser)
        # noinspection PyUnresolvedReferences
        for action in test_arg_parser._actions:
            if action.default != argparse.SUPPRESS:
                if getattr(options, action.dest, None) is None:
                    default_logger.info(
                        "Add missing option: {}={}".format(action.dest, action.default))
                    setattr(options, action.dest, action.default)


@six.add_metaclass(ABCMeta)
class GraphParserBase(DependencyParserBase):
    available_data_formats = {"sdp2014": graph_utils.Graph, "sdp2015": graph_utils.Graph2015}
    default_data_format_name = "sdp2014"


@six.add_metaclass(ABCMeta)
class TreeParserBase(DependencyParserBase):
    available_data_formats = {"conllu": tree_utils.Sentence}
    default_data_format_name = "conllu"
