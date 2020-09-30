import numpy as np
from . import cifar10
from . import svhn

class Cifar5Svhn(cifar10.Cifar5):

    def __init__(self, n_svhn=5000, *args, **kwargs):
        super(Cifar5Svhn, self).__init__(*args, **kwargs)
        self._n_svhn = n_svhn
        self._add_svhn_to_test()

    def _add_svhn_to_test(self):
        svhn_data = svhn.Svhn(test_only=True)
        x = svhn_data.test_x[:self._n_svhn]
        y = np.zeros(self._n_svhn, dtype=np.int64)
        y[:] = -2
        x = np.concatenate(
                [self.test_x, x], axis=0)
        y = np.concatenate(
                [self.test_y, y], axis=0)
        idx = self._rand.permutation(y.shape[0])
        self.test_x = x[idx]
        self.test_y = y[idx]

