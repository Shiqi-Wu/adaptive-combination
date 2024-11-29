import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 动态添加 utils 文件夹到 sys.path
current_dir = os.getcwd()
utils_dir = os.path.join(current_dir, "utils")
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

# 导入必要的模块
from rearrange_data import load_dataset, cut_slices, data_preparation, preprocess_data
from args_arguments import parse_arguments, read_config_file


def plot_column_distributions(x_data):
    """
    Plot the distribution of each column in the given data.

    Parameters:
    x_data (torch.Tensor or np.ndarray): Input data with columns to plot.

    Returns:
    None
    """
    if isinstance(x_data, torch.Tensor):
        x_data = x_data.cpu().numpy()  # Convert to NumPy array if it's a tensor

    # Determine the number of columns in the data
    num_columns = x_data.shape[1]

    # Create a histogram for each column
    plt.figure(figsize=(15, 5 * num_columns))
    for i in range(num_columns):
        column_data = x_data[:, i]  # Extract the data for the current column
        plt.subplot(num_columns, 1, i + 1)
        plt.hist(column_data, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of Column {i+1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data directories
data_dir = "data_dir/data_20240621"
config_dir = "outputs/experiment8/config.yaml"

# Parse config and load data
config = read_config_file(config_dir)
x_data, y_data, u_data = data_preparation(config, data_dir)

# Convert data to tensors and move to device
x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
u_data = torch.tensor(u_data, dtype=torch.float32).to(device)

# Preprocess the data
dataset = preprocess_data(x_data, y_data, u_data, device, config)

# Plot the distribution of each column in the input data
plot_column_distributions(x_data.cpu().numpy())  # Ensure data is on CPU for visualization