import os
import sys
import collections
from urllib import request
import numpy as np
from scipy import io

from . import data_base
from . import utils


DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
DATA_DIR = '/media/yw4/hdd/datasets/svhn'


class DataCache:
    """Avoid loading data mutiple times."""

    def __init__(self):
        self.train = None
        self.test = None


DATA_CACHE = DataCache()


def maybe_download_and_extract(data_url, data_dir=DATA_DIR):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading {} {:.1f}'
                    .format(filename, 
                        count * block_size / total_size * 100.0))
            sys.stdout.flush()
        filepath, _ = request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Sucessfully downloaded {} {} bytes'
                .format(filename, statinfo.st_size))


def load_train(data_dir=DATA_DIR, cache=DATA_CACHE, save_cache=True):
    maybe_download_and_extract(data_url=DATA_URL_TRAIN, data_dir=data_dir)
    data_file = os.path.join(data_dir, 'train_32x32.mat')
    if cache.train is None:
        data_mat = io.loadmat(data_file)
        x = data_mat['X'].transpose([3, 0, 1, 2])
        y = data_mat['y'].squeeze().astype(np.int64)
        y[y == 10] = 0
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
    maybe_download_and_extract(data_url=DATA_URL_TEST, data_dir=data_dir)
    data_file = os.path.join(data_dir, 'test_32x32.mat')
    if cache.test is None:
        data_mat = io.loadmat(data_file)
        x = data_mat['X'].transpose([3, 0, 1, 2])
        y = data_mat['y'].squeeze().astype(np.int64)
        y[y == 10] = 0
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


class Svhn(data_base.Dataset):

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


class Svhn5(data_base.Dataset):

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


class Svhn5TestBatch(data_base.Batch):

    def __init__(self, data_dir=DATA_DIR, n=5000, name='svhn5'):
        x_test_full, y_test_full = load_test(
                data_dir=data_dir, save_cache=False)
        classes = CLASSES1
        x_test, y_test = utils.subsample(
                x=x_test_full,
                y=y_test_full,
                classes=classes,
                n_samples=n,
                )
        data_dict = collections.OrderedDict()
        data_dict['x'] = (x_test, x_prepro)
        data_dict['y'] = (y_test, y_prepro)
        info_dict = collections.OrderedDict()
        info_dict['name'] = name
        info_dict['n_classes'] = len(classes)
        super().__init__(data_dict=data_dict, info_dict=info_dict)


class Svhn52TestBatch(data_base.Batch):

    def __init__(self, data_dir=DATA_DIR, n=5000, name='svhn52'):
        x_test_full, y_test_full = load_test(
                data_dir=data_dir, save_cache=False)
        classes = CLASSES2
        x_test, y_test = utils.subsample(
                x=x_test_full,
                y=y_test_full,
                classes=classes,
                n_samples=n,
                )
        data_dict = collections.OrderedDict()
        data_dict['x'] = (x_test, x_prepro)
        data_dict['y'] = (y_test, y_prepro)
        info_dict = collections.OrderedDict()
        info_dict['name'] = name
        info_dict['n_classes'] = len(classes)
        super().__init__(data_dict=data_dict, info_dict=info_dict)
