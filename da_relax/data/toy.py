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
    ns = (ns_p / np.sum(ns_p) * n_samples).astype(np.int32)
    ns[-1] = n_samples - np.sum(ns[:-1])
    for mean, std, n, label in zip(
            config['means'], 
            config['stds'], 
            ns, 
            config['labels']):
        x = sample_gaussian(np.array(mean), np.array(std), n, rand)
        y = np.zeros(n, dtype=np.int32)
        y[:] = label
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class Toy



class MixGsDataset(object):

    def __init__(self, 
            data_cfg, 
            n_samples,
            rand=None,
            shuffle=True):
        x, y = generate_xy(data_cfg, n_samples, rand)
        if shuffle:
            if rand is None:
                rand = np.random
            idx = rand.permutation(n_samples)
            x = x[idx]
            y = y[idx]
        self.x_ = x.astype(np.float32)
        self.y_ = y.astype(np.int32)
        self.n_samples = n_samples
        self.tf_prepro = {}
    
    def x(self):
        return self.x_

    def y(self):
        return self.y_



class MixGs(object):

    def __init__(self, 
            data_id, 
            n_train, 
            n_test, 
            seed=None,
            shuffle=True,
            path=None,
            ):

        data_cfg = data_cfg_factory(data_id)
        if seed is None:
            self.seed = np.random.randint(1000)
        else:
            self.seed = seed
        self.rand = np.random.RandomState(self.seed)
        self.n_train = n_train
        self.n_test = n_test
        self.shuffle = shuffle

        self.train = MixGsDataset(
                data_cfg, 
                n_samples=self.n_train,
                rand=self.rand,
                shuffle=self.shuffle
                )
        self.test = MixGsDataset(
                data_cfg,
                n_samples=self.n_test,
                rand=self.rand,
                shuffle=self.shuffle,
                )