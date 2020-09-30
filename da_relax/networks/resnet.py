import torch
import torch.nn as nn


def conv1x1(n_in, n_out, stride=1):
    return nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False)


def conv3x3(n_in, n_out, stride=1):
    return nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1,
            bias=False)


class ResidualBlock(nn.Module):

    def __init__(self, n_in, n_out, stride=1):
        super(ResidualBlock, self).__init__()
        self._stride = stride
        self.conv1 = conv3x3(n_in, n_out, stride=stride)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(n_out, n_out, stride=1)
        self.bn2 = nn.BatchNorm2d(n_out)
        if stride > 1:
            self.downsample = conv1x1(n_in, n_out, stride=stride)
            self.bn_downsample = nn.BatchNorm2d(n_out)

    def forward(self, x):
        identity = x
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        if self._stride > 1:
            identity = self.downsample(identity)
            identity = self.bn_downsample(identity)
        return self.relu(h + identity)


class ResNet18(nn.Module):
    """ResNet18 for 32x32 input."""

    def __init__(self, n_classes=10):
        super(ResNet18, self).__init__()
        self._n_classes = n_classes
        # layer configuration
        first_layer = (32, 1)  # n_out, stride
        res_blocks = (
                (32, 3, 1),
                (64, 3, 2),
                (128, 2, 2),
                )  # n_out, repeat, first block stride
        # first conv layer
        n_out = first_layer[0]
        self.conv = conv3x3(3, n_out, stride=1)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU()
        # residual blocks
        self.blocks = nn.ModuleList()
        for block_cfg in res_blocks:
            for i in range(block_cfg[1]):
                n_in = n_out
                n_out = block_cfg[0]
                if i == 0:
                    b = ResidualBlock(n_in, n_out, stride=block_cfg[2])
                else:
                    b = ResidualBlock(n_in, n_out, stride=1)
                self.blocks.append(b)
        # final output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(res_blocks[-1][0], n_classes)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        features = []
        features.append(x)
        h = x.permute(0, 3, 1, 2)
        h = self.conv(h)
        h = self.bn(h)
        h = self.relu(h)
        features.append(h)
        for b in self.blocks:
            h = b(h)
            features.append(h)
        # h = self.avgpool(h)
        # h = torch.flatten(h, 1)
        h = h.mean([2, 3])
        features.append(h)
        logits = self.fc_out(h)
        features.append(logits)
        return logits, features


