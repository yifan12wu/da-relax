import gzip
import os
from urllib import request
from urllib.parse import urljoin
import sys

import numpy as np
import collections

from . import data as data_lib
from . import utils


DATA_URL = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'
DATA_FILES = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz',
        }
DATA_DIR = '/media/yw4/hdd/datasets/usps'
# DATA_DIR = os.path.join(os.getcwd(), 'datasets/usps')

class DataCache:
    """Avoid loading data more than once."""

    def __init__(self):
        self.train = None
        self.test = None


DATA_CACHE = DataCache()


def _read_datafile(path):
    """Utility function for reading usps data files."""
    labels, images = [], []
    with gzip.GzipFile(path) as f:
        for line in f:
            vals = line.strip().split()
            labels.append(float(vals[0]))
            images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.int64)
        labels[labels == 10] = 0
        images = np.array(images, dtype=np.float32).reshape([-1, 16, 16, 1])
        images = (images + 1) / 2
    return images, labels


def maybe_download(data_dir=DATA_DIR):
    """Download usps dataset."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for filename in DATA_FILES.values():
        filepath = os.path.join(data_dir, filename)
        fileurl = os.path.join(DATA_URL, filename)
        # download if the file doesn't exist.
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading {} {:.1f}'
                        .format(filename, count*block_size/total_size*100.0))
                sys.stdout.flush()
            filepath, _ = request.urlretrieve(
                    fileurl, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Sucessfully downloaded {} {} bytes'
                    .format(filename, statinfo.st_size))


def load_train(data_dir=DATA_DIR, cache=DATA_CACHE, save_cache=True):
    if cache.train is None:
        filepath = os.path.join(data_dir, DATA_FILES['train'])
        x, y = _read_datafile(filepath)
        if save_cache:
            cache.train = (x, y)
    else:
        x, y = cache.train[0], cache.train[1]
    return x, y


def load_test(data_dir=DATA_DIR, cache=DATA_CACHE, save_cache=True):
    if cache.test is None:
        filepath = os.path.join(data_dir, DATA_FILES['test'])
        x, y = _read_datafile(filepath)
        if save_cache:
            cache.test = (x, y)
    else:
        x, y = cache.test[0], cache.test[1]
    return x, y


def x_prepro(x):
    return x


def y_prepro(y):
    return y


class USPS(data_lib.Dataset):
    """7291 train, 2007 test."""

    def __init__(self, 
            data_dir=DATA_DIR, 
            n_train=2000, 
            n_valid=None, 
            seed=0):
        maybe_download(data_dir)
        train = load_train(data_dir=data_dir)
        n = train[0].shape[0]
        if n_valid is None:
            n_valid = n - n_train
        split_sizes = [n_train, n_valid]
        train_valid = utils.subsample(
            batch=train, sizes=split_sizes, seed=seed)
        test = load_test(data_dir=data_dir)
        self._batches = {
            'train': train_valid[0],
            'valid': train_valid[1],
            'test': test,
        } 
        self._build()

    def _get_batch_keys(self):
        return ['train', 'valid', 'test']

    def _get_var_keys(self):
        return ['x', 'y']

    def _get_prepros(self):
        return [x_prepro, y_prepro]

    def _get_batch(self, key):
        return self._batches[key]

    def _get_info_dict(self):
        return {'n_classes': 10}


class SubsampledUSPS(USPS):

    def __init__(self,
            classes=(0, 1, 2, 3, 4), 
            data_dir=DATA_DIR, 
            n_train=2000, 
            n_valid=None, 
            seed=0):
        maybe_download(data_dir)
        train = load_train(data_dir=data_dir)
        test = load_test(data_dir=data_dir)
        # select all samples from the given classes 
        train = utils.select_classes(train, classes)
        test = utils.select_classes(test, classes) 
        # train valid split
        n = train[0].shape[0]
        if n_valid is None:
            n_valid = n - n_train
        split_sizes = [n_train, n_valid]
        train_valid = utils.subsample(
            batch=train, sizes=split_sizes, seed=seed)
        self._batches = {
            'train': train_valid[0],
            'valid': train_valid[1],
            'test': test,
        } 
        self._build()




