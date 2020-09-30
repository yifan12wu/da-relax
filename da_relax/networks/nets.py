from . import lenet
from . import resnet
from . import utils


def get_model(model_name, **kwargs):
    if model_name == 'lenet':
        model_cls = lenet.LeNet
    elif model_name == 'resnet18':
        model_cls = resnet.ResNet18
    else:
        raise ValueError('Unknown model {}.'.format(model_name))
    return utils.ModelWrapper(model_cls, **kwargs)

