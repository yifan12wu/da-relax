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


def get_div_fn(name, relax):
    return NotImplementedError