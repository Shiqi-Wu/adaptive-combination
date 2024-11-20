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
from loss_function import orthogonalize_columns, loss_function_orth
import matplotlib.pyplot as plt
import numpy as np

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set the random seed for reproducibility
torch.manual_seed(19)

dataset = generate_x_data(3000)

save_dir = 'outputs/experiment1'

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

def train(model, optimizer, data_loader, epochs, lr_decay_step = 50, lr_decay_gamma = 0.9):
    loss_history = []

    # Create a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    for epoch in range(epochs):
        # Train one epoch
        average_loss = train_one_epoch(model, optimizer, data_loader, epoch)
        loss_history.append(average_loss)

        # Step the scheduler
        scheduler.step()

        # Print average loss and current learning rate for the current epoch
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.6f}, Learning Rate: {current_lr:.6f}")

    return loss_history

if __name__ == "__main__":
    # Define the model
    model = ResNet(BasicBlock, [2, 2, 2]).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create the data loader
    data_loader = create_orthogonalized_data_loader(dataset, model, batch_size=128)

    # Train the model
    epochs = 4000
    loss_history = train(model, optimizer, data_loader, epochs)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet_model.pth"))
    np.save(os.path.join(save_dir, "loss_history.npy"), np.array(loss_history))
    print("Training complete. Model saved as resnet_model.pth")

    # plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "loss_history.png"))
    print("Loss history plot saved as loss_history.png")
    

    