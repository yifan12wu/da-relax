"""Inspect representations at intermediate layers."""
import os
import logging
import argparse

import numpy as np

from predprob.pytorch.tools import flag_tools
from predprob.pytorch.tools import logging_tools

from predprob.pytorch.eval_uncertainty import model_loader
from predprob.pytorch.data import datasets
from predprob.pytorch.networks import nets

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

FLAGS = parser.parse_args()
KB_EPS = 1e-5


def get_dists(x, y):
    # get cross distances between x and y
    x_norms = np.sum(np.square(x), axis=-1)
    y_norms = np.sum(np.square(y), axis=-1)
    inner_prods = x @ (y.T)
    dists = x_norms[:, None] + y_norms[None, :] - 2 * inner_prods
    return dists


def get_quantiles(x):
    quantiles = []
    for i in range(11):
        quantiles.append(i / 10)
    quantiles = np.array(quantiles)
    return np.quantile(x, quantiles)


def main():
    layer = FLAGS.layer
    kb_quantile = FLAGS.kb_quantile
    kb_multiplier = FLAGS.kb_multiplier
    batch_size = FLAGS.batch_size
    FLAGS.log_dir = os.path.join(
            FLAGS.log_dir,
            FLAGS.train_dataset,
            FLAGS.model,
            FLAGS.config_file,
            FLAGS.sub_dir)
    if not os.path.exists(FLAGS.log_dir):
        raise ValueError('log dir {} does not exist.'.format(FLAGS.log_dir))
    log_file = 'log_inspect_repr_{}.txt'.format(FLAGS.test_batch)
    logging_tools.config_logging(FLAGS.log_dir, filename=log_file)
    model_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
    logging.info('Loading dataset {} used for training...'
            .format(FLAGS.train_dataset))
    train_dataset = datasets.get_dataset(FLAGS.train_dataset)
    train_data = train_dataset.train
    # test_data_iid = train_dataset.test
    test_data_iid = datasets.get_test_batch(FLAGS.train_dataset)
    logging.info('Loading out-of-support dataset {}...'
            .format(FLAGS.test_batch))
    test_data_oos = datasets.get_test_batch(FLAGS.test_batch)
    n_classes = train_dataset.info.n_classes
    model = nets.get_model(FLAGS.model, n_classes=n_classes)
    logging.info('Loading model {} from {}...'
            .format(FLAGS.model, model_path))
    model_loader_ = model_loader.ModelLoader(
            model=model,
            model_path=model_path,
            batch_size=batch_size)
    # get training and test features
    logging.info('Getting feature representaions...')
    train_features, _, _ = model_loader_.get_outputs(
            data=train_data, layer=layer)
    test_features_iid, _, _ = model_loader_.get_outputs(
            data=test_data_iid, layer=layer)
    test_features_oos, _, _ = model_loader_.get_outputs(
            data=test_data_oos, layer=layer)
    logging.info(train_features.shape)
    logging.info(test_features_iid.shape)
    logging.info(test_features_oos.shape)
    logging.info('Getting pairwise distances...')
    tetr_dists_iid = get_dists(test_features_iid, train_features)
    tetr_dists_oos = get_dists(test_features_oos, train_features)
    trtr_dists = get_dists(train_features, train_features)
    # kernel bandwidth
    kb = np.quantile(tetr_dists_iid, kb_quantile)
    kb *= kb_multiplier
    kb = max(kb, KB_EPS)
    logging.info('Selected kernel bandwidth: {:.4g}.'.format(kb))
    te_kdes_iid = np.mean(np.exp(- tetr_dists_iid / kb), axis=-1)
    te_kdes_oos = np.mean(np.exp(- tetr_dists_oos / kb), axis=-1)
    tr_kdes = np.mean(np.exp(- trtr_dists / kb), axis=-1)
    logging.info('Training data kdes:')
    logging.info(get_quantiles(tr_kdes))
    logging.info('Test kdes iid data:')
    logging.info(get_quantiles(te_kdes_iid))
    logging.info('Test kdes oos data:')
    logging.info(get_quantiles(te_kdes_oos))


if __name__ == '__main__':
    main()
