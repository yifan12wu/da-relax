"""Test out of sample detection."""
import os
import argparse

from predprob.pytorch.tools import flag_tools
from predprob.pytorch.tools import logging_tools

import predprob.eval_uncertainty import kde_evaluator
from predprob.pytorch.data import datasets
from predprob.pytorch.networks import nets

################### flags ################
parser = argparse.ArgumentParser()
type_bool = flag_tools.type_bool

parser.add_argument('--log_dir', type=str,
        default='/media/yw4/hdd/log/uncertainty')
parser.add_argument('--train_dataset', type=str, default='cifar5')
parser.add_argument('--test_batch', type=str, default='svhn52')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--config_file', type=str, default='resnet_mm')
parser.add_argument('--sub_dir', type=str, default='test')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--kb_quantile', type=float, default=0.1)
parser.add_argument('--kb_multiplier', type=float, default=1.0)
parser.add_argument('--layer', type=int, default=-1)

#parser.add_argument('--sub_dir', type=str, default='0')
#parser.add_argument('--model_name', type=str, default='model')
#parser.add_argument('--config_file', type=str, default='cifar10_config')
#parser.add_argument('--dataset', type=str, default='cifar5svhn')
#parser.add_argument('--network', type=str, default='resnet18')
#parser.add_argument('--batch_size', type=int, default=128)
#parser.add_argument('--percentile', type=float, default=10)
#parser.add_argument('--kernel_variance', type=float, default=None)
#parser.add_argument('--layer', type=int, default=3)
#parser.add_argument('--mode', type=str, default='neg')

FLAGS = parser.parse_args()


def main():
    FLAGS.log_dir = os.path.join(
            FLAGS.log_dir,
            FLAGS.train_dataset,
            FLAGS.model,
            FLAGS.config_file,
            FLAGS.sub_dir)
    if not os.path.exists(FLAGS.log_dir):
        raise ValueError('log dir {} does not exist'
                .format(FLAGS.log_dir))
    logging_tools.config_logging(FLAGS.log_dir)
    evaluator_ = kde_evaluator.Evaluator(
            model_factory=model_factory,
            dataset_factory=dataset_factory,
            percentile=FLAGS.percentile,
            kernel_variance=FLAGS.kernel_variance,
            layer=FLAGS.layer,
            mode=FLAGS.mode,
            flags=FLAGS)

    evaluator_.evaluate()


if __name__ == '__main__':
    main()
