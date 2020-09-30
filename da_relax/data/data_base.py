"""In-memory dataset interface."""
import collections
import numpy as np
from ..tools import flag_tools


def one_shot_index_iterator(n, batch_size):
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        indices = np.arange(batch_size) + start
        if end < start + batch_size:
            indices[end - start:] = 0
        effective_size = end - start
        yield indices, effective_size
        start += batch_size


def random_index_iterator(n, batch_size):
    while True:
        indices = np.random.choice(n, batch_size, replace=False)
        yield indices, batch_size


class Batch:

    def __init__(self, data_dict, info_dict=None):
        """
        data_dict: {key: (val, prepro)}.
        """
        size = None
        data_keys = []
        data = flag_tools.Flags()
        info = flag_tools.Flags()
        for key, (val, prepro) in data_dict.items():
            if size is None:
                size = val.shape[0]
            else:
                assert size == val.shape[0]
            if prepro is None:
                def _identity(x):
                    return x
                prepro = _identity
            dummy_batch = prepro(val[:1])
            shape = dummy_batch.shape[1:]
            item = flag_tools.Flags()
            item.val = val
            item.prepro = prepro
            item.shape = shape
            setattr(data, key, item)
            data_keys.append(key)
        if info_dict is not None:
            for key, val in info_dict.items():
                setattr(info, key, val)
        self._data = data
        self._info = info
        self._data_keys = data_keys
        self._size = size

    @property
    def data_keys(self):
        return self._data_keys

    @property
    def size(self):
        return self._size

    @property
    def data(self):
        return self._data

    @property
    def info(self):
        return self._info

    def _data_iterator(self, index_iterator):
        for batch_indices, size in index_iterator:
            minibatch = flag_tools.Flags()
            for key in self._data_keys:
                full_batch = getattr(self._data, key)
                sampled_batch = full_batch.val[batch_indices].copy()
                sampled_batch = full_batch.prepro(sampled_batch)
                setattr(minibatch, key, sampled_batch)
            yield minibatch, size

    def get_random_iterator(self, batch_size):
        batch_size = min(batch_size, self._size)
        index_iterator = random_index_iterator(
                self._size, batch_size)
        return self._data_iterator(index_iterator)

    def get_one_shot_iterator(self, batch_size):
        batch_size = min(batch_size, self._size)
        index_iterator = one_shot_index_iterator(
                self._size, batch_size)
        return self._data_iterator(index_iterator)


class Dataset:

    def _build(self):
        info_dict = self._get_info_dict()
        keys = self._get_keys()
        prepros = self._get_prepros()
        train_data = self._get_train()
        if train_data is not None:
            data_dict = self._get_data_dict(keys, train_data, prepros)
            self._train = Batch(data_dict=data_dict, info_dict=info_dict)
        else:
            self._train = None
        valid_data = self._get_valid()
        if valid_data is not None:
            data_dict = self._get_data_dict(keys, valid_data, prepros)
            self._valid = Batch(data_dict=data_dict, info_dict=info_dict)
        else:
            self._valid = None
        test_data = self._get_test()
        if test_data is not None:
            data_dict = self._get_data_dict(keys, test_data, prepros)
            self._test = Batch(data_dict=data_dict, info_dict=info_dict)
        else:
            self._test = None
        info = flag_tools.Flags()
        if info_dict is not None:
            for key, val in info_dict.items():
                setattr(info, key, val)
        self._info = info

    def _get_data_dict(self, keys, data, prepros):
        data_dict = collections.OrderedDict()
        for key, d, prepro in zip(keys, data, prepros):
            data_dict[key] = (d, prepro)
        return data_dict

    def _get_keys(self):
        raise NotImplementedError

    def _get_train(self):
        return None

    def _get_valid(self):
        return None

    def _get_test(self):
        return None

    def _get_prepros(self):
        keys = self._get_keys()
        return list([None for _ in keys])

    def _get_info_dict(self):
        return {}

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test

    @property
    def info(self):
        return self._info


