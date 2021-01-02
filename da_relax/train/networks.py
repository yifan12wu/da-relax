import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet for 16x16x1 input."""

    def __init__(self, n_classes):
        super(LeNet, self).__init__()
        self._n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # size-2 pooling on outputs of each conv layer
        # 8x8 image size on outputs of conv2
        n_in = 4 * 4 * 64
        self.fc1 = nn.Linear(n_in, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self._n_classes)

    def forward(self, x):
        features = []
        features.append(x)
        h = x.permute(0, 3, 1, 2)
        # conv layer 1
        h = self.conv1(h)
        h = F.relu(h)
        features.append(h)
        h = F.avg_pool2d(h, 2)
        # conv layer 2
        h = self.conv2(h)
        h = F.relu(h)
        features.append(h)
        h = F.avg_pool2d(h, 2)
        # fc layers
        h = h.reshape(h.shape[0], -1)
        h = F.relu(self.fc1(h))
        features.append(h)
        h = F.relu(self.fc2(h))
        features.append(h)
        h = self.fc3(h)
        logits = h
        features.append(logits)
        return logits, features


class MLP(nn.Module):

    def __init__(self, 
            input_shape, 
            n_units, 
            output_activation=False,
            ):
        super().__init__()
        self._layers = []
        n_in = int(np.prod(np.array(input_shape)))
        for i, n_out in enumerate(n_units):
            layer = nn.Linear(n_in, n_out)
            self.add_module('hidden_layer_{}'.format(i+1), layer)
            n_in = n_out
            self._layers.append(layer)
        self._output_activation = output_activation

    def forward(self, x):
        features = []
        h = x.reshape(x.shape[0], -1)
        features.append(h)
        for layer in self._layers[:-1]:
            h = layer(h)
            h = F.relu(h)
            features.append(h)
        h = self._layers[-1](h)
        if self._output_activation:
            h = F.relu(h)
        features.append(h)
        return h, features





