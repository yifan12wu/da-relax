import gzip
import operator
import os
import struct
from functools import reduce
from urllib import request
from urllib.parse import urljoin
import sys

import numpy as np
import collections

from . import data as data_lib
from . import utils


DATA_URL = 'http://yann.lecun.com/exdb/mnist'
DATA_FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
        }
DATA_DIR = '/media/yw4/hdd/datasets/mnist'
# DATA_DIR = os.path.join(os.getcwd(), 'datasets/mnist')


class DataCache:
    """Avoid loading data more than once."""

    def __init__(self):
        self.train = None
        self.test = None


DATA_CACHE = DataCache()


def _read_datafile(path, expected_dims):
    """Utility function for reading mnist data files."""
    base_magic_num = 2048
    with gzip.GzipFile(path) as f:
        magic_num = struct.unpack('>I', f.read(4))[0]
        expected_magic_num = base_magic_num + expected_dims
        if magic_num != expected_magic_num:
            raise ValueError('Incorrect MNIST magic number (expected '
                    '{}, got {})'.format(expected_magic_num, magic_num))
        dims = struct.unpack('>' + 'I' * expected_dims,
                f.read(4 * expected_dims))
        buf = f.read(reduce(operator.mul, dims))
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(dims)
        return data


def _read_images(path):
    """Read an mnist image file, return as an NHWC np array."""
    return _read_datafile(path, 3).reshape([-1, 28, 28, 1])


def _read_labels(path):
    """Read an mnist label file, return as an np array with size [N]."""
    return _read_datafile(path, 1)


def maybe_download(data_dir=DATA_DIR):
    """Download mnist dataset."""
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
        image_path = os.path.join(data_dir, DATA_FILES['train_images'])
        label_path = os.path.join(data_dir, DATA_FILES['train_labels'])
        x = _read_images(image_path)
        y = _read_labels(label_path)
        if save_cache:
            cache.train = (x, y)
    else:
        x, y = cache.train[0], cache.train[1]
    return x, y


def load_test(data_dir=DATA_DIR, cache=DATA_CACHE, save_cache=True):
    if cache.test is None:
        image_path = os.path.join(data_dir, DATA_FILES['test_images'])
        label_path = os.path.join(data_dir, DATA_FILES['test_labels'])
        x = _read_images(image_path)
        y = _read_labels(label_path)
        if save_cache:
            cache.test = (x, y)
    else:
        x, y = cache.test[0], cache.test[1]
    return x, y


def x_prepro(x):
    return x.astype(np.float32) / 255.0


def y_prepro(y):
    return y.astype(np.int64)


class MNIST(data_lib.Dataset):
    """60000 train, 10000 test."""

    def __init__(self, 
            data_dir=DATA_DIR, 
            n_train=50000, 
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


class SubsampledMNIST(MNIST):

    def __init__(self,
            classes=(0, 1, 2, 3, 4), 
            data_dir=DATA_DIR, 
            n_train=10000, 
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




