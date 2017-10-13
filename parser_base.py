import random

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
from common_utils import set_proc_name, add_common_arguments, add_train_arguments, add_predict_arguments, ensure_dir, \
    add_train_and_predict_arguments
from logger import logger, log_to_file
from training_scheduler import TrainingScheduler


@six.add_metaclass(ABCMeta)
class DependencyParserBase(object):
    DataType = abstractproperty()

    @abstractmethod
    def train(self, graphs):
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
        group.add_argument("--epochs", type=int, dest="epochs", default=30)
        group.add_argument("--lr", type=float, dest="learning_rate", default=0.1)

    @classmethod
    def train_parser(cls, options, data_train=None, data_dev=None, data_test=None):
        set_proc_name(options.title)
        ensure_dir(options.output)
        path = os.path.join(options.output, "{}_{}_train.log".format(options.title,
                                                                     int(time.time())))
        log_to_file(path)
        logger.name = options.title

        logger.info('Options:\n%s', pformat(options.__dict__))
        if data_train is None:
            data_train = cls.DataType.from_file(options.conll_train)

        if data_dev is None:
            data_dev = {i: cls.DataType.from_file(i, False) for i in options.conll_dev}

        try:
            os.makedirs(options.output)
        except OSError:
            pass

        parser = cls(options, data_train)
        random_obj = random.Random(1)
        for epoch in range(options.epochs):
            logger.info('Starting epoch %d', epoch)
            random_obj.shuffle(data_train)
            parser.train(data_train)

            # save model and delete old model
            for i in range(0, epoch - options.max_save):
                path = os.path.join(options.output, os.path.basename(options.model)) + str(i + 1)
                if os.path.exists(path):
                    os.remove(path)
            path = os.path.join(options.output, os.path.basename(options.model)) + str(epoch + 1)
            parser.save(path)

            def predict(sentences, gold_file, output_file):
                with open(output_file, "w") as f_output:
                    for i in parser.predict(sentences):
                        f_output.write(i.to_string())
                # script_path = os.path.join(os.path.dirname(__file__), "main.py")
                # p = subprocess.Popen([sys.executable, script_path, "mst+empty", "predict", "--model", path,
                #                       "--test", gold_file,
                #                       "--output", output_file], stdout=sys.stdout)
                # p.wait()
                cls.DataType.evaluate_with_external_program(gold_file, output_file)

            for file_name, file_content in data_dev.items():
                try:
                    prefix, suffix = os.path.basename(file_name).rsplit(".", 1)
                except ValueError:
                    prefix = file_name
                    suffix = ""

                dev_output = os.path.join(options.output, '{}_epoch_{}.{}'.format(prefix, epoch + 1, suffix))
                predict(file_content, file_name, dev_output)

    @classmethod
    def predict_with_parser(cls, options):
        data_test = cls.DataType.from_file(options.conll_test, False)

        logger.info('Initializing...')
        parser = cls.load(options.model, options)

        ts = time.time()
        with open(options.out_file, "w") as f_output:
            for i in parser.predict(data_test):
                f_output.write(i.to_string())
        te = time.time()
        logger.info('Finished predicting and writing test. %.2f seconds.', te - ts)

    @classmethod
    def get_arg_parser(cls):
        parser = ArgumentParser(sys.argv[0])
        cls.fill_arg_parser(parser)
        return parser

    @classmethod
    def fill_arg_parser(cls, parser):
        add_common_arguments(parser)
        sub_parsers = parser.add_subparsers()
        sub_parsers.required = True
        sub_parsers.dest = 'mode'

        # Train
        train_subparser = sub_parsers.add_parser("train")
        add_train_arguments(train_subparser)
        add_train_and_predict_arguments(train_subparser)
        cls.add_parser_arguments(train_subparser)
        train_subparser.set_defaults(func=cls.train_parser)

        # Predict
        predict_subparser = sub_parsers.add_parser("predict")
        add_predict_arguments(predict_subparser)
        add_train_and_predict_arguments(predict_subparser)
        predict_subparser.set_defaults(func=cls.predict_with_parser)

    @classmethod
    def get_training_scheduler(cls, train=None, dev=None, test=None):
        return TrainingScheduler(cls.train_parser, cls.get_arg_parser(), train, dev ,test)


@six.add_metaclass(ABCMeta)
class GraphParserBase(DependencyParserBase):
    DataType = graph_utils.Graph


@six.add_metaclass(ABCMeta)
class TreeParserBase(DependencyParserBase):
    DataType = tree_utils.Sentence
