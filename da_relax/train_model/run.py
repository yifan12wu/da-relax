import os

import argparse
import shutil
import importlib
import logging

from predprob.pytorch.tools import flag_tools
from predprob.pytorch.tools import logging_tools
from predprob.pytorch.tools import datetime_tools

from predprob.pytorch.train_model import trainer

# -- flags
parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str,
        default='/media/yw4/hdd/log/uncertainty')
parser.add_argument('--sub_dir', type=str, default='test')
parser.add_argument('--config_dir', type=str, 
        default='predprob.pytorch.train_model.config')
parser.add_argument('--config_file', type=str, default='resnet_mm')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--refresh_dir', type=flag_tools.type_bool, default=True)
parser.add_argument('--args', type=str, action='append', default=[])

FLAGS = parser.parse_args()


def main():
    if FLAGS.sub_dir == 'auto':
        FLAGS.sub_dir = datetime_tools.get_time()
    FLAGS.log_dir = os.path.join(
            FLAGS.log_dir,
            FLAGS.dataset,
            FLAGS.model,
            FLAGS.config_file,
            FLAGS.sub_dir)
    if FLAGS.refresh_dir:
        if os.path.exists(FLAGS.log_dir):
            shutil.rmtree(FLAGS.log_dir)
    else:
        FLAGS.log_dir = flag_tools.get_unique_dir(FLAGS.log_dir)
    os.makedirs(FLAGS.log_dir)
    logging_tools.config_logging(FLAGS.log_dir)
    # load config
    config_module = importlib.import_module('{}.{}'.format(
        FLAGS.config_dir, FLAGS.config_file))
    cfg = config_module.Config(FLAGS)
    logging.info(cfg.flags_dict)
    cfg.save_flags(log_dir=FLAGS.log_dir)
    # train
    trainer_ = trainer.Trainer(**cfg.trainer_args)
    trainer_.train()


if __name__ == '__main__':
    main()

