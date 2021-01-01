import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputHead:

    def __init__(self, logits, features):
        self.logits = logits
        self.features = features
        self.predprobs = F.softmax(self.logits, -1)
        self.predictions = torch.argmax(self.logits, 1)

    def get_label(self, y):
        self.losses = F.cross_entropy(
                input=self.logits, target=y, reduction='none')
        self.loss = self.losses.mean()
        self.accs = (self.predictions == y).to(dtype=torch.float32)
        self.acc = self.accs.mean()


class ModelWrapper(nn.Module):

    def __init__(self, model_cls, **kwargs):
        super(ModelWrapper, self).__init__()
        self.model = model_cls(**kwargs)

    def forward(self, x, is_training=True):
        self.model.train(is_training)
        logits, features = self.model(x)
        return OutputHead(logits, features)


def soft_relu(x):
    """Compute log(1 + exp(x)) with numerical stability.
    
    Can be used for getting differentiable nonnegative outputs.
    Might also be useful in other cases, e.g.:  
        log(sigmoid(x)) = x - soft_relu(x) = - soft_relu(-x).
        log(1 - sigmoid(x)) = - soft_relu(x)
    """
    return ((-x.abs()).exp() + 1.0).log() + torch.clip(x, min=0.0)


def js_div(d1, d2):
    """Jensen-Shannon divergence given dual function outputs.
    
    Return:
        E[log(sigmoid(d1))] + E[log(1 - sigmoid(d2))] + log(4)
    """
    part1 = - soft_relu(-d1).mean()
    part2 = - soft_relu(d2).mean() 
    return part1 + part2 + np.log(4.0)


def js_beta(d1, d2, beta):
    """Relaxed Jensen-Shannon divergence given dual function outputs.

    minimizing js_beta gives p2/p1 <= 1 + beta.
    Return:
        E[log(sigmoid(d1))] + E[log(2 + beta - sigmoid(d2))] - log(1 + beta)
    """
    part1 = - soft_relu(-d1).mean()
    part2 = (beta + 2.0 - (- soft_relu(-d2)).exp()).log().mean()
    return part1 + part2 - np.log(1.0 + beta)


def wasserstein_beta(d1, d2, beta):
    """Relaxed Wasserstein distance given dual function outputs.

    minimizing wasserstein_beta gives p2/p1 <= 1 + beta.
    Return:
        E[log(sigmoid(d1))] + E[log(2 + beta - sigmoid(d2))] - log(1 + beta)
    """
    part1 = - soft_relu(-d1).mean()
    part2 = (beta + 2.0 - (- soft_relu(-d2)).exp()).log().mean()
    return part1 + part2 - np.log(1.0 + beta)


def js_sort(d1, d2, beta):
    """Reweighting-Relaxed Jensen-Shannon divergence.

    minimizing js_sort gives p2/p1 <= 1 + beta.
    Return:
        E[log(sigmoid(d1))] + E[log(2 + beta - sigmoid(d2))] - log(1 + beta)
    """
    part1 = - soft_relu(-d1).mean()
    part2 = (beta + 2.0 - (- soft_relu(-d2)).exp()).log().mean()
    return part1 + part2 - np.log(1.0 + beta)


def get_div_fn(name, relax=0.0):
    assert relax >= 0
    if name == 'js':
        if relax > 0:
            raise ValueError('Use js_beta instead of js when relax > 0.')
        _div_fn = js_div
    elif name == 'js_beta':
        def _div_fn(d1, d2):
            return js_beta(d1, d2, relax)
    elif name == 'w_beta':
        def _div_fn(d1, d2):
            return wasserstein_beta(d1, d2, relax)
    elif name == 'jssort':
        def _div_fn(d1, d2):
            return js_sort(d1, d2, relax)
    else:
        raise ValueError('Unknown divergence function {}.'.format(name))
    return _div_fn