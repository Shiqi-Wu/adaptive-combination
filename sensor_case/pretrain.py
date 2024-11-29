import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))
from model import BasicBlock, ResNet, generate_x_data
from model import Linear_model
from loss_function import orthogonalize_columns, loss_function_orth, loss_function_fit, loss_function_total
from args_arguments import parse_arguments, read_config_file
import matplotlib.pyplot as plt
import numpy as np

from rearrange_data import load_dataset, cut_slices, data_preparation, preprocess_data
import numpy as np
from model import StdScalerLayer, PCALayer, Rescale_pca_layer
from train import linear_regression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_orthogonalized_data_loader(dataset, model, batch_size):
    def collate_fn(batch):
        # Unpack the batch data
        x = torch.stack(batch).to(device)

        # Convert lists of samples into tensors
        x = x.to(device)
        ones = torch.ones(x.size(0), 1).to(device)

        # Concatenate the ones column with x
        inputs = torch.cat((ones, x), dim=1)

        # Generate `g` using the model and orthogonalize it
        with torch.no_grad():
            g = x
            orthogonal_g = orthogonalize_columns(g)

        # Return processed batch data
        return inputs, orthogonal_g

    # Return DataLoader with custom collate function
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def train_one_epoch(model, optimizer, data_loader, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Progress bar for current epoch
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}", leave=True)

    for inputs, orthogonal_g in progress_bar:
        optimizer.zero_grad()

        # Compute `h` using the model's parameter `k`
        h = model(inputs)

        # Compute the loss
        loss = loss_function_orth(orthogonal_g, h)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    # Return average loss for the epoch
    average_loss = total_loss / num_batches
    return average_loss

def test_one_epoch(model, val_loader):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            inputs, orthogonal_g = batch
            h = model(inputs)
            loss = loss_function_orth(orthogonal_g, h)
            total_loss += loss.item()

    average_loss = total_loss / num_batches
    return average_loss

def train(model, optimizer, data_loader, val_data_loader, epochs, lr_decay_step = 300, lr_decay_gamma = 0.9):
    loss_history = []
    val_loss_history = []

    # Create a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    for epoch in range(epochs):
        # Train one epoch
        average_loss = train_one_epoch(model, optimizer, data_loader, epoch)
        loss_history.append(average_loss)

        val_loss = test_one_epoch(model, val_data_loader)
        val_loss_history.append(val_loss)

        # Step the scheduler
        scheduler.step()

        # Print average loss and current learning rate for the current epoch
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.6f}, Val Average Loss: {val_loss:.6f}, Learning Rate: {current_lr:.6f}")

    return loss_history, val_loss_history


def main():

    # Define the model
    model = ResNet(BasicBlock, [2, 2, 2]).to(device)

    # set the random seed for reproducibility
    torch.manual_seed(19)

    save_dir = 'outputs/experiment6'
    os.makedirs(save_dir, exist_ok=True)

    dataset = generate_x_data(3000)
    val_dataset = generate_x_data(1000)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = ResNet(BasicBlock, [2, 2, 2]).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create the data loader
    data_loader = create_orthogonalized_data_loader(dataset, model, batch_size=128)
    val_data_loader = create_orthogonalized_data_loader(val_dataset, model, batch_size=128)

    # Training configuration
    epochs = 20000
    os.makedirs(save_dir, exist_ok=True)

    # Train the model
    loss_history, val_loss_history = train(model, optimizer, data_loader, val_data_loader, epochs)

    # Save the trained model and loss histories
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet_model.pth"))
    np.save(os.path.join(save_dir, "loss_history.npy"), np.array(loss_history))
    np.save(os.path.join(save_dir, "val_loss_history.npy"), np.array(val_loss_history))
    print("Training complete. Model saved as resnet_model.pth")

    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.yscale('log')
    plt.title("Training Loss History")
    y_ticks = [5e-2, 8e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1e0]
    plt.yticks(y_ticks, [f"{y:.0e}" for y in y_ticks])
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "loss_history.png"))
    print("Loss history plot saved as loss_history.png")


def create_orthogonalized_data_loader_v2(dataset, batch_size):
    def collate_fn(batch):
        # Unpack the batch data
        x, g, e = zip(*batch)

        x = torch.stack(x)  # 将列表或元组的 x 转换为张量
        g = torch.stack(g)  # 同样转换 g
        e = torch.stack(e)

        # Convert lists of samples into tensors
        x = x.to(device)
        g = g.to(device)
        e = e.to(device)
        ones = torch.ones(g.size(0), 1).to(device)

        # Concatenate the ones column with x
        inputs = torch.cat((ones, x), dim=1)

        # Generate `g` using the model and orthogonalize it
        with torch.no_grad():
            orthogonal_g = orthogonalize_columns(g)

        # Return processed batch data
        return inputs, orthogonal_g, e

    # Return DataLoader with custom collate function
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def train_one_epoch_v2(model, optimizer, data_loader, epoch):
    model.train()
    total_loss = 0.0
    total_loss_fit = 0.0
    total_loss_orth = 0.0
    num_batches = 0

    # Progress bar for current epoch
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}", leave=True)

    for inputs, orthogonal_g, e in progress_bar:
        optimizer.zero_grad()

        # Compute `h` using the model's parameter `k`
        h = model(inputs)

        # Compute individual losses
        loss_orth = loss_function_orth(orthogonal_g, h)
        loss_fit = loss_function_fit(e, h)
        loss = loss_function_total(orthogonal_g, h, e)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_loss_fit += loss_fit.item()
        total_loss_orth += loss_orth.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    # Return average losses for the epoch
    average_loss = total_loss / num_batches
    average_loss_fit = total_loss_fit / num_batches
    average_loss_orth = total_loss_orth / num_batches
    return average_loss, average_loss_fit, average_loss_orth

def test_one_epoch_v2(model, val_loader):
    model.eval()
    total_loss = 0.0
    total_loss_fit = 0.0
    total_loss_orth = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for inputs, orthogonal_g, e in val_loader:
            h = model(inputs)

            # Compute individual losses
            loss_orth = loss_function_orth(orthogonal_g, h)
            loss_fit = loss_function_fit(e, h)
            loss = loss_function_total(orthogonal_g, h, e)

            # Accumulate losses
            total_loss += loss.item()
            total_loss_fit += loss_fit.item()
            total_loss_orth += loss_orth.item()

    average_loss = total_loss / num_batches
    average_loss_fit = total_loss_fit / num_batches
    average_loss_orth = total_loss_orth / num_batches
    return average_loss, average_loss_fit, average_loss_orth

def train_v2(model, optimizer, data_loader, val_data_loader, epochs, lr_decay_step=300, lr_decay_gamma=0.9):
    loss_history = []
    loss_fit_history = []
    loss_orth_history = []
    val_loss_history = []
    val_loss_fit_history = []
    val_loss_orth_history = []

    # Create a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    for epoch in range(epochs):
        # Train one epoch
        average_loss, average_loss_fit, average_loss_orth = train_one_epoch_v2(model, optimizer, data_loader, epoch)
        loss_history.append(average_loss)
        loss_fit_history.append(average_loss_fit)
        loss_orth_history.append(average_loss_orth)

        # Validate one epoch
        val_loss, val_loss_fit, val_loss_orth = test_one_epoch_v2(model, val_data_loader)
        val_loss_history.append(val_loss)
        val_loss_fit_history.append(val_loss_fit)
        val_loss_orth_history.append(val_loss_orth)

        # Step the scheduler
        scheduler.step()

        # Print average losses and current learning rate for the current epoch
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.6f}, Fit Loss: {average_loss_fit:.6f}, "
              f"Orth Loss: {average_loss_orth:.6f}, Val Average Loss: {val_loss:.6f}, "
              f"Val Fit Loss: {val_loss_fit:.6f}, Val Orth Loss: {val_loss_orth:.6f}, Learning Rate: {current_lr:.6f}")

    return loss_history, loss_fit_history, loss_orth_history, val_loss_history, val_loss_fit_history, val_loss_orth_history

def main_v2():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = ResNet(BasicBlock, [2, 2, 2]).to(device)

    # Parse command-line arguments
    args = parse_arguments()
    config = read_config_file(args.config)

    # Load data
    data_dir = config['train_data_dir']
    x_data, y_data, u_data = data_preparation(config, data_dir)
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    u_data = torch.tensor(u_data, dtype=torch.float32).to(device)

    # Preprocess the data
    dataset, rescale_pca_layer = preprocess_data(x_data, y_data, u_data, device, config)


    # Define the model
    x_data, y_data, u_data = dataset
    state_dim = config['pca_dim'] + 1
    control_dim = dataset[2].shape[1]
    linear_model = Linear_model(state_dim, control_dim).to(device)

    linear_model = linear_regression(dataset, linear_model, device)

    linear_predict = linear_model(x_data, u_data).detach()
    err_data = y_data - linear_predict

    training_dataset = [(x, linear_predict[i], err_data[i]) for i, x in enumerate(x_data)]
    # Define the data loader
    data_loader = create_orthogonalized_data_loader_v2(training_dataset, batch_size=128)

    # Define the optimizer
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.001)

    # Training configuration
    epochs = config['pretrain_epochs']
    save_dir = config['pretrain_save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Train the model
    loss_history, loss_fit_history, loss_orth_history, val_loss_history, val_loss_fit_history, val_loss_orth_history = train_v2(model, optimizer, data_loader, data_loader, epochs)

    # Save the trained model and loss histories
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet_model.pth"))
    np.save(os.path.join(save_dir, "loss_history.npy"), np.array(loss_history))
    np.save(os.path.join(save_dir, "loss_fit_history.npy"), np.array(loss_fit_history))
    np.save(os.path.join(save_dir, "loss_orth_history.npy"), np.array(loss_orth_history))
    np.save(os.path.join(save_dir, "val_loss_history.npy"), np.array(val_loss_history))
    np.save(os.path.join(save_dir, "val_loss_fit_history.npy"), np.array(val_loss_fit_history))
    np.save(os.path.join(save_dir, "val_loss_orth_history.npy"), np.array(val_loss_orth_history))
    print("Training complete. Model saved as resnet_model.pth")
    
    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_loss_history, label="Validation Loss")
    plt.plot(range(1, epochs + 1), loss_fit_history, label="Training Fit Loss")
    plt.plot(range(1, epochs + 1), val_loss_fit_history, label="Validation Fit Loss")
    plt.plot(range(1, epochs + 1), loss_orth_history, label="Training Orth Loss")
    plt.plot(range(1, epochs + 1), val_loss_orth_history, label="Validation Orth Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.yscale('log')
    plt.title("Training Loss History")
    y_ticks = [5e-2, 8e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1e0]
    plt.yticks(y_ticks, [f"{y:.0e}" for y in y_ticks])
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "loss_history.png"))
    print("Loss history plot saved as loss_history.png")
    return
    


if __name__ == "__main__":
    main()