"""An in-memory dataset interface."""
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
    """
    A batch instance contains a (large) batch of data (e.g. training/validation
    batch) and can generate iterators for given mini-batch sizes.

    A batch instance provides the following access to the data batch:
        var_keys: a list of variable keys.
        size: number of samples.
        data: data.[var_key].(val/prepro/shape).
        info: info.[info_key].
        get_one_shot_iterator: return a one shot iterator given batch size.
        get_random_iterator: return a random iterator given batch size.
    """

    def __init__(self, data_dict, info_dict=None):
        """
        data_dict: {var_key: (val, prepro)}. A batch of data may contain
        multiple variables (e.g., x, y). var_key is the name of the
        variable. val is the (batched) value. prepro is the proprocessing
        needed when generating minibatches.

        info_dict contains additional information about the data batch.
        """
        size = None
        var_keys = []
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
            var_keys.append(key)
        if info_dict is not None:
            for key, val in info_dict.items():
                setattr(info, key, val)
        self._data = data
        self._info = info
        self._var_keys = var_keys
        self._size = size

    @property
    def var_keys(self):
        return self._var_keys

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
            for key in self._var_keys:
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
    """A dataset contains a set of (possibly overlapping) batches."""

    def _build(self):
        batch_keys = self._get_batch_keys()
        info_dict = self._get_info_dict()
        var_keys = self._get_var_keys()
        prepros = self._get_prepros()
        for batch_key in batch_keys:
            batch = self._get_batch(batch_key)
            data_dict = self._get_data_dict(var_keys, batch, prepros)
            if hasattr(self, batch_key):
                raise ValueError('Invalid batch key: {}.'.format(batch_key))
            else:
                batch = Batch(data_dict=data_dict, info_dict=info_dict)
                setattr(self, batch_key, batch)
        # convert info dict to flags
        info = flag_tools.Flags()
        if info_dict is not None:
            for key, val in info_dict.items():
                setattr(info, key, val)
        self._info = info

    def _get_data_dict(self, var_keys, data, prepros):
        data_dict = collections.OrderedDict()
        for key, d, prepro in zip(var_keys, data, prepros):
            data_dict[key] = (d, prepro)
        return data_dict

    def _get_batch_keys(self):
        return ['train', 'valid', 'test']

    def _get_var_keys(self):
        return ['x', 'y']

    def _get_prepros(self):
        """A list of proprocessing functions in the same order of var_keys."""
        keys = self._get_var_keys()
        identity_map = lambda x: x
        return [identity_map for _ in keys]

    def _get_batch(self, key):
        # return a list of data batches, in the order of var_keys
        raise NotImplementedError

    def _get_info_dict(self):
        return {}

    @property
    def batch_keys(self):
        return self._get_batch_keys()

    @property
    def info(self):
        return self._info


