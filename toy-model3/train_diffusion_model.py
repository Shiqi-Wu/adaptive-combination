import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SimpleModel_diffusion, diffusion_equation
from loss_function import loss_function_orth,  orthogonalize_columns

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate dataset
data_model = diffusion_equation()
dataset = data_model.generate_training_data(500, 10, dlt_t=0.001)

# Create a custom DataLoader with a collate function for batch processing
def create_orthogonalized_data_loader(dataset, model, batch_size):
    def collate_fn(batch):
        # Unpack the batch data
        x, y, lace_1, lace_2 = zip(*batch)

        # Convert lists of samples into tensors
        x = torch.stack(x).to(device)
        y = torch.stack(y).to(device)
        lace_1 = torch.stack(lace_1).to(device)
        lace_2 = torch.stack(lace_2).to(device)

        # Generate `g` using the model and orthogonalize it
        with torch.no_grad():
            g, _ = model(lace_1, lace_2)
            orthogonal_g = orthogonalize_columns(g)

        # Return processed batch data
        return x, y, orthogonal_g, lace_1, lace_2

    # Return DataLoader with custom collate function
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define training function for one epoch
def train_one_epoch(model, optimizer, data_loader, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Progress bar for current epoch
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}", leave=True)

    for x, y, orthogonal_g, lace_1, lace_2 in progress_bar:
        optimizer.zero_grad()

        # Compute `h` using the model's parameter `k`
        _, h = model(lace_1, lace_2)

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

# Define the main training function
def train(model, optimizer, data_loader, epochs):
    loss_history = []
    k_history = []

    for epoch in range(epochs):
        # Train one epoch
        average_loss = train_one_epoch(model, optimizer, data_loader, epoch)
        loss_history.append(average_loss)

        # Save the current value of `model.k`
        k_history.append(model.k.item())

        # Print average loss for the current epoch
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.6f}")

    return loss_history, k_history

if __name__ == "__main__":
    # Initialize model and move it to device
    model = SimpleModel_diffusion().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    batch_size = 64
    epochs = 10

    # Create DataLoader
    orthogonalized_data_loader = create_orthogonalized_data_loader(dataset, model, batch_size)

    # Train the model
    loss_history, k_history = train(model, optimizer, orthogonalized_data_loader, epochs)

    # Save loss_history and k_history to .npy files
    np.save("loss_history.npy", np.array(loss_history))
    np.save("k_history.npy", np.array(k_history))
    print("Training history saved to loss_history.npy and k_history.npy")

    # Plot and save loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid()
    plt.savefig("loss_history.png")
    print("Loss history plot saved as loss_history.png")

    # Plot and save k history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), k_history, label="Model k Value")
    plt.xlabel("Epoch")
    plt.ylabel("k Value")
    plt.title("Model k Value History")
    plt.legend()
    plt.grid()
    plt.savefig("k_history.png")
    print("k history plot saved as k_history.png")
