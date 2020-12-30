import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet for 32x32 input."""

    def __init__(self, n_classes=10):
        super(LeNet, self).__init__()
        self._n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # size-2 max pool on outputs of each conv layer
        # 8x8 image size on outputs of conv2
        n_in = 8 * 8 * 64
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



