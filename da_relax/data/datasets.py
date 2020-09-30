from . import cifar10
from . import svhn
from . import utils


def get_dataset(name, **kwargs):
    if name == 'cifar5':
        return cifar10.Cifar5(**kwargs)
    if name == 'cifar10':
        return cifar10.Cifar10(**kwargs)
    if name == 'svhn5':
        return svhn.Svhn5(**kwargs)
    if name == 'svhn':
        return svhn.Svhn(**kwargs)
    else:
        raise ValueError('Unknown dataset {}.'.format(name))


TEST_BATCHES = {
        'cifar5': cifar10.Cifar5TestBatch,
        'cifar52': cifar10.Cifar52TestBatch,
        'svhn5': svhn.Svhn5TestBatch,
        'svhn52': svhn.Svhn52TestBatch,
        }


def get_test_batch(name, **kwargs):
    return TEST_BATCHES[name](name=name, **kwargs)


def get_combined_test_batch(names, kwargs_list=None):
    batches = []
    if kwargs_list is None:
        for name in names:
            batches.append(get_test_batch(name))
    else:
        for name, kwargs in zip(names, kwargs_list):
            batches.append(get_test_batch(name, **kwargs))
    return utils.combine_batches(batches)

