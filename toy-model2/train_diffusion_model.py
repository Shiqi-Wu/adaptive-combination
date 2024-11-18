import numpy as np
import matplotlib.pyplot as plt
from model import *
import sys 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from loss_function import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import gradcheck
from torch.optim import LBFGS
from tqdm import tqdm

# Define a simple model with a single trainable parameter `k`
class SimpleModel(nn.Module):
    def __init__(self, k_value = None):
        super(SimpleModel, self).__init__()
        if k_value is None:
            self.k = nn.Parameter(torch.randn(()))
        else:
            self.k = nn.Parameter(torch.tensor(k_value))

    def forward(self, lace_1, lace_2):
        g = lace_1
        h = self.k * lace_1 + lace_2
        return g, h


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
            x, y, lace_1, lace_2 = batch
            g, h = model(lace_1, lace_2)
            loss = loss_fn(y - x, g, h)[0]
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.item()

        # Update progress bar and writer
        progress_bar.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4e}")
        
        if batch_idx % 10 == 0:
            writer.add_scalar('Batch Loss', loss.item(), epoch * len(train_loader) + batch_idx)
            
            # Track gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad, epoch * len(train_loader) + batch_idx)
            
            writer.add_scalar('k value', model.k.item(), epoch * len(train_loader) + batch_idx)

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Training Epoch: {epoch + 1}, Average Loss: {avg_loss:.4e}, k: {model.k.item():.4e}")
    writer.add_scalar('Training Loss', avg_loss, epoch)
    return avg_loss

if __name__ == '__main__':
    data_model = diffusion_equation()
    dataset = data_model.generate_training_data(500, 10, dlt_t = 0.001)

    model = SimpleModel()

    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    for batch in data_loader:
        x, y, lace_1, lace_2 = batch
        break

    # Define the optimizer and scheduler
    optimizer = LBFGS(model.parameters(), lr=0.1)
    epochs = 5
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # Define the loss function
    loss_fn = loss_function


    writer = SummaryWriter()

    train_loss_history = []
    para_history = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, scheduler, data_loader, loss_fn, epoch, writer)
        train_loss_history.append(train_loss)
        para_history.append(model.k.clone().detach().numpy())

    # Exploring k values and computing the loss
    k_values = np.linspace(-4, 4, 100)
    terminal_losses = []
    orthogonal_losses = []

    for k in k_values:
        model = SimpleModel(k)
        g, h = model(lace_1, lace_2)
        _, loss_1, loss_2 = loss_function(y, g, h)
        # print(f'k: {k}, Terminal Loss: {loss_1.item():.4e}, Orthogonal Loss: {loss_2.item():.4e}')
        terminal_losses.append(loss_1.item())
        orthogonal_losses.append(loss_2.item())

    # Log losses to TensorBoard
    for k, term_loss, ortho_loss in zip(k_values, terminal_losses, orthogonal_losses):
        writer.add_scalars('Losses_on_k', {'Terminal Loss': term_loss, 'Orthogonal Loss': ortho_loss}, k)

    writer.close()