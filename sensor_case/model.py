import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

# Define a function to generate random data
def generate_x_data(size):
    """
    Generates x_data from a standard normal distribution N(0,1) with shape (size, 4).

    Args:
    size (int): The number of rows in the generated tensor.

    Returns:
    torch.Tensor: A tensor of shape (size, 4) with values drawn from N(0,1).
    """
    return torch.randn(size, 4)

# Define the ResNet model
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

# Define a model with a pretrained ResNet and a fully connected layer
class PretrainedModelWithFC(nn.Module):
    def __init__(self, pretrained_model, num_pretrained, num_classes):
        super(PretrainedModelWithFC, self).__init__()
        self.pretrained_model = pretrained_model

        # Freeze the parameters of the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.norm = nn.BatchNorm1d(num_pretrained)

        # Add a fully connected layer
        self.fc = nn.Linear(number_pretrained, num_classes)

    def forward(self, x):
        # Forward pass through the pretrained model
        with torch.no_grad():
            x = self.pretrained_model(x)

        # Normalize the output
        x = self.norm(x)

        # Forward pass through the fully connected layer
        x = self.fc(x)
        return x

# Define a linear model
class Linear_model(nn.Module):
    def __init__(self, state_dim, control_dim):
        super(Linear_model, self).__init__()
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(control_dim, state_dim))
    
    def x_dict(self, x):
        ones = torch.ones(x.shape[0], 1).to(x.device)
        return torch.cat((ones, x), dim=1)

    def forward(self, x, u):
        x = self.x_dict(x)
        y = torch.matmul(x, self.A) + torch.matmul(u, self.B)
        return y[:, 1:]


# Define the data rescaling layers
class PCALayer(nn.Module):
    def __init__(self, input_dim, output_dim, pca_matrix):
        super(PCALayer, self).__init__()
        self.pca_matrix = pca_matrix
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = nn.Linear(input_dim, output_dim, bias = False)
        self.transform.weight = nn.Parameter(pca_matrix, requires_grad=False)
        self.inverse_transform = nn.Linear(output_dim, input_dim, bias = False)
        self.inverse_transform.weight = nn.Parameter(pca_matrix.T, requires_grad=False)

class StdScalerLayer(nn.Module):
    def __init__(self, mean, std):
        super(StdScalerLayer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, input):
        return input * self.std + self.mean

class Rescale_pca_layer(nn.Module):
    def __init__(self, std_layer_1, std_layer_2, std_layer_u, pca_layer):
        super(Rescale_pca_layer, self).__init__()
        self.std_layer_1 = std_layer_1
        self.std_layer_2 = std_layer_2
        self.std_layer_u = std_layer_u
        self.pca_layer = pca_layer

    def transform_x(self, x):
        x = self.std_layer_1.transform(x)
        x = self.pca_layer.transform(x)
        x = self.std_layer_2.transform(x)
        return x
    
    def inverse_transform_x(self, x):
        x = self.std_layer_2.inverse_transform(x)
        x = self.pca_layer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x)
        return x

    def transform_u(self, u):
        return self.std_layer_u.transform(u)

    def inverse_transform_u(self, u):
        return self.std_layer_u.inverse_transform(u)