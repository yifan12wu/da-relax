import numpy as np

from . import data as data_lib


def sample_gaussian(mean, std, n, rand=None):
    if rand is None:
        rand = np.random
    x = rand.normal(size=[n, mean.shape[0]]) * std + mean
    return x


DATA_CONFIG = {
    'source': {
        'means': [[-1., -0.3], [1., 0.3]],
        'stds': [[0.1, 0.4], [0.1, 0.4]],
        'ns': [5, 5],
        'labels': [0, 1],
    },
    'target': {
        'means': [[-0.3, -1.0], [0.3, 1.0]],
        'stds': [[0.4, 0.1], [0.4, 0.1]],
        'ns': [1, 9],
        'labels': [0, 1],
    },
}


def generate_xy(config, n_samples, rand=None):
    xs = []
    ys = []
    ns_p = np.array(config['ns']).astype(np.float32)
    ns = (ns_p / np.sum(ns_p) * n_samples).astype(np.int64)
    ns[-1] = n_samples - np.sum(ns[:-1])
    for mean, std, n, label in zip(
            config['means'], 
            config['stds'], 
            ns, 
            config['labels']):
        x = sample_gaussian(np.array(mean), np.array(std), n, rand)
        y = np.zeros(n, dtype=np.int64)
        y[:] = label
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class ToyDataset(data_lib.Dataset):

    def __init__(self,
            config,
            n_train=1000,
            n_valid=1000,
            n_test=1000,
            seed=0):
        rand = np.random.RandomState(seed)
        train = generate_xy(config, n_train, rand)
        valid = generate_xy(config, n_valid, rand)
        test = generate_xy(config, n_test, rand)
        self._batches = {'train': train, 'valid': valid, 'test': test}
        self._config = config
        self._build()

    def _get_batch_keys(self):
        return ['train', 'valid', 'test']

    def _get_var_keys(self):
        return ['x', 'y']

    def _get_batch(self, key):
        return self._batches[key]

    def _get_info_dict(self):
        return {'n_classes': len(self._config['labels'])}



class TwoGaussianSource(ToyDataset):

    def __init__(self,
            n_train=1000,
            n_valid=1000,
            n_test=1000,
            seed=0):
        super().__init__(
            config=DATA_CONFIG['source'],
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            seed=seed)


class TwoGaussianTarget(ToyDataset):

    def __init__(self,
            n_train=1000,
            n_valid=1000,
            n_test=1000,
            seed=1):
        super().__init__(
            config=DATA_CONFIG['target'],
            n_train=n_train,
            n_valid=n_valid,
            n_test=n_test,
            seed=seed)