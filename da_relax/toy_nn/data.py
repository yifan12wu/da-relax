# 1d-image datasets
import numpy as np
from ..data import data_base


def identity_fn(x):
    return x


class Base(data_base.Dataset):
    
    def __init__(self, dim, n_train, replace=True, seed=None):
        self._dim = dim
        self._n_train = n_train
        self.x, self.y = self._gen_data()
        self._n = self.x.shape[0]
        rand = np.random.RandomState(seed)
        train_idx = rand.choice(self._n, size=self._n_train, replace=replace)
        self.x_train = self.x[train_idx]
        self.y_train = self.y[train_idx]
        self._build()

    def _gen_data(self):
        x = np.zeros([self._n_train * 2, self._dim])
        y = np.zeros([self._n_train * 2]).astype(np.int64)
        return x, y

    def _get_keys(self):
        return ('x', 'y')

    def _get_train(self):
        return (self.x_train, self.y_train)

    def _get_valid(self):
        return (self.x, self.y)

    def _get_test(self):
        return (self.x, self.y)

    def _get_prepros(self):
        return (identity_fn, identity_fn)

    def _get_info_dict(self):
        return {}


class Relative(Base):

    def _gen_data(self):
        # d * (d-1) samples, covering all cases
        d = self._dim
        x = np.zeros([d * (d - 1), d])
        y = np.zeros([d * (d - 1)]).astype(np.int64)
        idx = 0
        for i in range(d):
            neg = i
            for j in range(d - 1):
                pos = (neg + 1 + j) % d
                x[idx, neg] = -1
                x[idx, pos] = 1
                if neg < pos:
                    y[idx] = 1
                idx += 1
        return x, y


class Basis(Base):

    def __init__(self, dim, n_train, freq=1, **kwargs):
        """
        freq:
        an integer between 1 and dim indicating how frequent the label changes.
        e.g. 1 for all poistive labels, dim for 1-0-1-0-... like labels.
        -1 means random labels
        """
        self._freq = freq
        super().__init__(dim, n_train, **kwargs)

    def _gen_data(self):
        x = np.identity(self._dim)
        y = np.zeros([self._dim]).astype(np.int64)
        inv_len = int(self._dim // self._freq)
        st = 0
        while st < self._dim:
            y[st:st + inv_len] = 1
            st += inv_len * 2
        return x, y


def get_dataset(name, dim, n_train, seed):
    if name == 'cls':
        return Basis(dim=dim, n_train=n_train, seed=seed, freq=1)
    if name == 'ctrl1':
        return Basis(dim=dim, n_train=n_train, seed=seed, freq=2)
    if name == 'ctrl3':
        return Relative(dim=dim, n_train=n_train, seed=seed)
