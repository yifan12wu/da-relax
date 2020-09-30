import os
import argparse

from tools import flag_tools
from tools import logging_tools

import softmax_evaluator
from data import datasets
from networks import nets

################### flags ################
parser = argparse.ArgumentParser()
type_bool = flag_tools.type_bool

parser.add_argument('--log_dir', type=str,
        default='/media/yw4/hdd/log/predprob')
parser.add_argument('--sub_dir', type=str, default='final')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--config_file', type=str, default='cifar10_config')

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--network', type=str, default='lenet')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--temp', type=float, default=1.0)

FLAGS = parser.parse_args()

FLAGS.log_dir = os.path.join(
        FLAGS.log_dir,
        FLAGS.dataset,
        FLAGS.network,
        FLAGS.sub_dir)


def model_factory(config):
    return nets.get_model(FLAGS.network, FLAGS.dataset)

def dataset_factory(config):
    return datasets.get_np_dataset(FLAGS.dataset, FLAGS.batch_size)

def main():
    # refresh logdir
    if not os.path.exists(FLAGS.log_dir):
        raise ValueError('log dir {} does not exist'
                .format(FLAGS.log_dir))
    logging_tools.config_logging(FLAGS.log_dir)
    evaluator_ = softmax_evaluator.Evaluator(
            model_factory=model_factory,
            dataset_factory=dataset_factory,
            temp=FLAGS.temp,
            flags=FLAGS)

    evaluator_.evaluate()


if __name__ == '__main__':
    main()
