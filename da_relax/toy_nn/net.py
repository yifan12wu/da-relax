import torch
import torch.nn as nn
import torch.nn.functional as F


def identity_fn(x):
    return x


def relu_fn(x):
    return F.relu(x)


class ConvNet(nn.Module):

    def __init__(self, d_in, k, n_conv_layers, n_fc_layers, init_mul=1.0,
            activation='none', use_bias=False):
        super().__init__()
        assert int(k % 2) == 1  # filter size must be an odd number
        if activation == 'none':
            self.act_fn = identity_fn
        elif activation == 'relu':
            self.act_fn = relu_fn
        else:
            raise ValueError('unknown activation.')
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for _ in range(n_conv_layers):
            layer = nn.Conv1d(1, 1, kernel_size=k,
                    stride=1, padding=int((k - 1) / 2), bias=use_bias)
            self.conv_layers.append(layer)
        for _ in range(n_fc_layers):
            layer = nn.Linear(d_in, d_in, bias=use_bias)
            self.fc_layers.append(layer)
        self.output_layer = nn.Linear(d_in, 1, bias=use_bias)
        # self.output_layer.weight.data.zero_()
        if init_mul != 1.0:
            for w in self.weights:
                w.data.copy_(w.data * init_mul)

    def forward(self, x):
        b = x.shape[0]
        h = x.reshape(b, 1, -1)
        for l in self.conv_layers:
            h = self.act_fn(l(h))
        h = h.reshape(b, -1)
        for l in self.fc_layers:
            h = self.act_fn(l(h))
        o = self.output_layer(h)
        return o.reshape(-1)

    @property
    def weights(self):
        w_list = []
        for l in self.conv_layers:
            w_list.append(l.weight)
        for l in self.fc_layers:
            w_list.append(l.weight)
        w_list.append(self.output_layer.weight)
        return w_list




class OutputHead:

    def __init__(self, logits, loss='ce'):
        self.logits = logits
        self._loss_name = loss
        if loss == 'ce' or loss == 'linear':
            self.predictions = (logits > 0).to(dtype=torch.int64)
        elif loss == 'l2':
            self.predictions = (logits > 0.5).to(dtype=torch.int64)

    def get_label(self, y):
        y_float = y.to(dtype=torch.float32)
        if self._loss_name == 'ce':
            self.losses = F.binary_cross_entropy_with_logits(
                    input=self.logits, target=y_float, reduction='none')
        if self._loss_name == 'linear':
            self.losses = - self.logits * y_float
        elif self._loss_name == 'l2':
            self.losses = F.mse_loss(
                    input=self.logits, target=y_float, reduction='none')
        self.loss = self.losses.mean()
        self.accs = (self.predictions == y).to(dtype=torch.float32)
        self.acc = self.accs.mean()


class ModelWrapper(nn.Module):

    def __init__(self, loss='ce', **kwargs):
        super().__init__()
        self.model = ConvNet(**kwargs)
        self.loss = loss

    def forward(self, x, is_training=True):
        self.model.train(is_training)
        logits = self.model(x)
        return OutputHead(logits, loss=self.loss)



