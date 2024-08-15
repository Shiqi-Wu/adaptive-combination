import numpy as np
import matplotlib.pyplot as plt
import model
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from loss_function import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import gradcheck
from torch.optim import LBFGS
from tqdm import tqdm

class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

class SimpleModel(nn.Module):
    def __init__(self, params, NN_params, NN_state_dict = None):
        super(SimpleModel, self).__init__()
        self.params = params
        self.NN_params = NN_params
        self.NN = TwoLayerNN(NN_params.input_size, NN_params.hidden_size, NN_params.output_size)
        if NN_state_dict is not None:
            self.NN.load_state_dict(NN_state_dict)

    def forward(self, x, lace):
        g = lace
        h = self.NN(x)
        return g, h
    
    def set_params(self, params):
        self.params = params

    def forward_y(self, x, lace):
        g, h = self.forward(x, lace)
        return x + self.params.k * g + torch.matmul(h, self.params.linear_layer)

lambda_1, lambda_2 = 1, 1
def loss_function(y, g, h):
    loss_1 = terminal_loss_ver2(g, h, y)
    loss_2 = orthogonal_loss_ver2(g, h)
    return lambda_1 * loss_1 + lambda_2 * loss_2, loss_1, loss_2


def train_one_epoch(model, optimizer, scheduler, train_loader, loss_fn, epoch, writer):
    model.train()  # Set model to training mode
    total_loss = 0.0

    # Create progress bar
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

    # Iterate over batches
    for batch_idx, batch in progress_bar:
        def closure():
            optimizer.zero_grad()  # Clear gradients
            x, y, lace = batch
            g, h = model(x, lace)
            loss = loss_fn(y, g, h)[0]
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.item()

        progress_bar.set_description(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Training Epoch: {epoch}, Average Loss: {avg_loss:.4f}")
    if writer is not None:
        writer.add_scalar("Loss/train_avg", avg_loss, epoch)   

    return avg_loss

def test_one_epoch(model, test_loader, loss_fn, epoch, writer):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    # Create progress bar
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))

    # Iterate over batches
    for batch_idx, batch in progress_bar:
        x, y, lace = batch
        g, h = model(x, lace)
        loss = loss_fn(y, g, h)[0]
        total_loss += loss.item()

        progress_bar.set_description(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(test_loader)
    print(f"Test Epoch: {epoch}, Average Loss: {avg_loss:.4f}")
    if writer is not None:
        writer.add_scalar("Loss/test_avg", avg_loss, epoch)

    return avg_loss

class Reaction_diffusion_param():
    def __init__(self, k = 0, linear_layer = None):
        self.k = torch.tensor(k, requires_grad = False)
        # torch.random.manual_seed(0)
        if linear_layer is None:
            self.linear_layer = torch.zeros(10, 1, requires_grad = False)
        else:
            self.linear_layer = linear_layer

class NN_param():
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

if __name__ == '__main__':
    data_model = model.reaction_diffusion_equation()

    params = Reaction_diffusion_param(1)
    NN_params = NN_param(1, 32, 10)
    model = SimpleModel(params, NN_params)

    # Generate training data
    train_dataset = torch.load('train_dataset.pth')
    test_dataset = torch.load('test_dataset.pth')


    # train_dataset = data_model.generate_training_data(1000, 50)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    # test_dataset = data_model.generate_training_data(200, 50)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    # torch.save(train_dataset, 'train_dataset.pth')
    # torch.save(test_dataset, 'test_dataset.pth')
    # Set up optimizer and scheduler
    optimizer = Adam(model.NN.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    epochs = 50
    writer = SummaryWriter()


    # # Define the loss function
    # lambda_1, lambda_2 = 1e6, 0
    # loss_fn = loss_function

    # for epoch in range(epochs):
    #     train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, loss_fn, epoch, writer)
    #     test_loss = test_one_epoch(model, test_loader, loss_fn, epoch, writer)
    #     writer.add_scalar("Loss/train_avg", train_loss, epoch)
    #     writer.add_scalar("Loss/test_avg", test_loss, epoch)

    # ############################################################
    # Expreiment 2
    lambda_1, lambda_2 = 1e6, 1
    loss_fn = loss_function
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, loss_fn, epoch, writer)
        test_loss = test_one_epoch(model, test_loader, loss_fn, epoch, writer)
        writer.add_scalar("Loss/train_avg", train_loss, epoch)
        writer.add_scalar("Loss/test_avg", test_loss, epoch)
    writer.close()

    # Save the model
    torch.save(model.state_dict(), "model_pretrain.pth")


