import torch
from torch import nn
from torch.nn import functional as F

def generate_x_data(size):
    """
    Generates x_data from a standard normal distribution N(0,1) with shape (size, 4).

    Args:
    size (int): The number of rows in the generated tensor.

    Returns:
    torch.Tensor: A tensor of shape (size, 4) with values drawn from N(0,1).
    """
    return torch.randn(size, 4)

class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=64):
        super(ResNet, self).__init__()
        self.in_features = 5

        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1])
        self.layer3 = self._make_layer(block, 64, num_blocks[2])
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_features, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_features, out_features))
            self.in_features = out_features
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.linear(out)
        return out

def ResNet3():
    return ResNet(BasicBlock, [1, 1, 1])