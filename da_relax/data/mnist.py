import gzip
import operator
import os
import struct
from functools import reduce
from uillib.parse import urljoin

import numpy as np
import collections

from . import data_base
from . import utils


DATA_URL = 'http://yann.lecun.com/exdb/mnist'
DATA_FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
        }
DATA_DIR = '/media/yw4/hdd/datasets/mnist'


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
    if cache.train is not None:
        image_path = os.path.join(data_dir, DATA_FILES['train_images'])
        label_path = os.path.join(data_dir, DATA_FILES['train_labels'])
        x = _read_images(image_path)
        y = _read_labels(label_path)
        # shuffle
        rand = np.random.RandomState(0)
        idx = rand.permutation(x.shape[0])
        x = x[idx]
        y = y[idx]
        if save_cache:
            cache.train = (x, y)
    else:
        x, y = cache.train[0], cache.train[1]
    return x, y


def load_test(data_dir=DATA_DIR, cache=DATA_CACHE, save_cache=True):
    if cache.test is not None:
        image_path = os.path.join(data_dir, DATA_FILES['test_images'])
        label_path = os.path.join(data_dir, DATA_FILES['test_labels'])
        x = _read_images(image_path)
        y = _read_labels(label_path)
        # shuffle
        rand = np.random.RandomState(0)
        idx = rand.permutation(x.shape[0])
        x = x[idx]
        y = y[idx]
        if save_cache:
            cache.test = (x, y)
    else:
        x, y = cache.test[0], cache.test[1]
    return x, y


def x_prepro(x):
    return x.astype(np.float32) / 255.0


def y_prepro(y):
    return y.astype(np.int64)


class Cifar10(data_base.Dataset):

    def __init__(self, data_dir=DATA_DIR, n_train=40000):
        self._data_dir = data_dir
        self._n_train = n_train
        x_train, y_train = load_train(data_dir=data_dir)
        self._x_test, self._y_test = load_test(data_dir=data_dir)
        self._x_train = x_train[:self._n_train]
        self._y_train = y_train[:self._n_train]
        self._x_valid = x_train[self._n_train:]
        self._y_valid = y_train[self._n_train:]
        self._build()

    def _get_keys(self):
        return ('x', 'y')

    def _get_train(self):
        return (self._x_train, self._y_train)

    def _get_valid(self):
        return (self._x_valid, self._y_valid)

    def _get_test(self):
        return (self._x_test, self._y_test)

    def _get_prepros(self):
        return (x_prepro, y_prepro)

    def _get_info_dict(self):
        return {'n_classes': 10}


CLASSES1 = [0, 1, 2, 3, 4]
CLASSES2 = [5, 6, 7, 8, 9]


class Cifar5(data_base.Dataset):

    def __init__(self, data_dir=DATA_DIR, n_train=10000, n_valid=5000):
        assert n_train <= 20000
        x_train_full, y_train_full = load_train(
                data_dir=data_dir, save_cache=False)
        x_test_full, y_test_full = load_test(
                data_dir=data_dir, save_cache=False)
        self._n_train = n_train
        self._n_valid = n_valid
        self._classes = CLASSES1
        # training and validation
        train_and_valid = utils.subsample(
                x=x_train_full, 
                y=y_train_full,
                classes=self._classes,
                n_samples=[n_train, n_valid])
        self._x_train, self._y_train = train_and_valid[0]
        self._x_valid, self._y_valid = train_and_valid[1]
        # test
        self._x_test, self._y_test = utils.subsample(
                x=x_test_full,
                y=y_test_full,
                classes=self._classes)
        self._build()

    def _get_keys(self):
        return ('x', 'y')

    def _get_train(self):
        return (self._x_train, self._y_train)

    def _get_valid(self):
        return (self._x_valid, self._y_valid)

    def _get_test(self):
        return (self._x_test, self._y_test)

    def _get_prepros(self):
        return (x_prepro, y_prepro)

    def _get_info_dict(self):
        return {'n_classes': len(self._classes)}

        
class Cifar5TestBatch(data_base.Batch):

    def __init__(self, data_dir=DATA_DIR, name='cifar5'):
        x_test_full, y_test_full = load_test(
                data_dir=data_dir, save_cache=False)
        classes = CLASSES1
        x_test, y_test = utils.subsample(
                x=x_test_full,
                y=y_test_full,
                classes=classes)
        data_dict = collections.OrderedDict()
        data_dict['x'] = (x_test, x_prepro)
        data_dict['y'] = (y_test, y_prepro)
        info_dict = collections.OrderedDict()
        info_dict['name'] = name
        info_dict['n_classes'] = len(classes)
        super().__init__(data_dict=data_dict, info_dict=info_dict)


class Cifar52TestBatch(data_base.Batch):

    def __init__(self, data_dir=DATA_DIR, name='cifar52'):
        x_test_full, y_test_full = load_test(
                data_dir=data_dir, save_cache=False)
        classes = CLASSES2
        x_test, y_test = utils.subsample(
                x=x_test_full,
                y=y_test_full,
                classes=classes)
        data_dict = collections.OrderedDict()
        data_dict['x'] = (x_test, x_prepro)
        data_dict['y'] = (y_test, y_prepro)
        info_dict = collections.OrderedDict()
        info_dict['name'] = name
        info_dict['n_classes'] = len(classes)
        super().__init__(data_dict=data_dict, info_dict=info_dict)

