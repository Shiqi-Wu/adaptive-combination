import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import yaml
from sklearn.decomposition import PCA
import copy
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from rearrange_data import load_dataset, cut_slices, data_preparation
from model import Linear_model, PretrainedModelWithFC, Rescale_pca_layer, StdScalerLayer, PCALayer

# Parses command-line arguments to obtain configuration settings.
def parse_arguments():
    """
    Input:
        None
    Function:
        Parses command-line arguments to retrieve the path to the configuration file.
        If no path is provided, defaults to 'config.yaml'.
    Returns:
        args (Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

# Reads and loads the configuration from a YAML file.
def read_config_file(config_file):
    """
    Input:
        config_file (str): Path to the YAML configuration file.
    Function:
        Opens the specified YAML file and loads its contents as a dictionary.
        If there is an error during reading, it will print the error message.
    Returns:
        config (dict): Configuration parameters loaded from the YAML file.
    """
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

# Performs linear regression to estimate parameters A and B for the linear model.
def linear_regression(dataset, linear_model, device='cpu'):
    """
    Inputs:
        dataset (tuple): A tuple containing tensors (x, y, u), where
            x: Input features.
            y: Target outputs.
            u: Control variables or additional inputs.
        linear_model (nn.Module): A PyTorch model with linear components.
        device (str): The computation device ('cpu' or 'cuda').
    Function:
        Performs linear regression to calculate parameters A and B of the linear model 
        by using pseudo-inverse computation.
    Returns:
        linear_model (nn.Module): The updated linear model with parameters A and B.
    """
    x, y, u = dataset
    x = x.to(device)
    y = y.to(device)
    u = u.to(device)
        
    x_dic = linear_model.x_dict(x)
    y_dic = linear_model.x_dict(y)

    z = torch.cat((x_dic, u), dim=1)
    z_pseudo_inv = torch.pinverse(z)

    param_pseudo_inv = torch.matmul(z_pseudo_inv, y_dic)
    A = param_pseudo_inv[:x_dic.shape[1], :]
    B = param_pseudo_inv[x_dic.shape[1]:, :]

    with torch.no_grad():
        linear_model.register_parameter('A', nn.Parameter(A))
        linear_model.register_parameter('B', nn.Parameter(B))
    
    return linear_model

# Trains the linear model by fitting it to the residuals of the data.
def train_linear_model(dataset, linear_model, residual_model, device='cpu'):
    """
    Inputs:
        dataset (tuple): A tuple containing tensors (x_data, y_data, u_data), where
            x_data: Input features.
            y_data: Target outputs.
            u_data: Control variables or additional inputs.
        linear_model (nn.Module): A PyTorch model with linear components.
        residual_model (nn.Module): A PyTorch model representing the residuals to be fitted.
        device (str): The computation device ('cpu' or 'cuda').
    Function:
        Adjusts the parameters of the linear model by minimizing the residual error
        between the target outputs and the residual model's predictions.
    Returns:
        linear_model (nn.Module): The updated linear model with adjusted parameters.
    """
    x_data, y_data, u_data = dataset
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    u_data = u_data.to(device)
    if residual_model is None:
        raise ValueError("The residual model must be provided.")
    err_data = y_data - residual_model(x_data, u_data)
    linear_model = linear_regression((x_data, err_data, u_data), linear_model, device)
    return linear_model

def hybrid_loss(linear_model, residual_model, x_data_seq, u_data_seq):
    """
    Inputs:
        linear_model (nn.Module): The linear component of the model.
        residual_model (nn.Module): The residual component of the model.
        x_data_seq (torch.Tensor): Input data sequence of shape (batch_size, sequence_length, feature_size).
        u_data_seq (torch.Tensor): Control input sequence of shape (batch_size, sequence_length, control_size).
    Function:
        Computes the mean squared error (MSE) loss between the predicted sequence and the actual data sequence.
        The prediction is generated by recursively applying the linear and/or residual models over the sequence.
    Returns:
        loss (torch.Tensor): The computed MSE loss.
    """
    mse = nn.MSELoss()
    N = x_data_seq.shape[1]
    if N <= 1:
        raise ValueError("The sequence length must be greater than 1.")
    
    # Move data to the appropriate device
    x_data_seq = x_data_seq.to(residual_model.device)
    u_data_seq = u_data_seq.to(residual_model.device)

    # Initialize the prediction with the first element of the sequence
    x0 = x_data_seq[:, 0, :]
    x_pred = [x0]

    # Iteratively predict the next elements in the sequence
    for i in range(1, N):
        if residual_model is not None and linear_model is not None:
            # Apply both linear and residual models
            x0 = linear_model(x0, u_data_seq[:, i-1, :]) + residual_model(x0, u_data_seq[:, i-1, :])
        elif linear_model is not None:
            # Apply only the linear model
            x0 = linear_model(x0, u_data_seq[:, i-1, :])
        else:
            # Apply only the residual model
            x0 = residual_model(x0, u_data_seq[:, i-1, :])
        x_pred.append(x0)
    
    # Stack the predicted sequence along the sequence dimension
    x_pred = torch.stack(x_pred, dim=1)
    # Compute the mean squared error loss between predictions and actual data
    loss = mse(x_pred, x_data_seq)
    return loss

def train_one_epoch(linear_model, residual_model, optimizer, data_loader, epoch):
    if linear_model is not None:
        linear_model.eval()
    
    if residual_model is None:
        raise ValueError("The residual model must be provided.")
    
    residual_model.train()
    
    total_loss = 0
    for x_data_seq, u_data_seq in tqdm(data_loader, desc = f"Epoch {epoch + 1}", unit = "batch"):
        optimizer.zero_grad()
        loss = hybrid_loss(linear_model, residual_model, x_data_seq, u_data_seq)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def test_one_epoch(linear_model, residual_model, data_loader):
    if linear_model is not None:
        linear_model.eval()
    
    if residual_model is None:
        raise ValueError("The residual model must be provided.")
    
    residual_model.eval()

    total_loss = 0
    for x_data_seq, u_data_seq in data_loader:
        loss = hybrid_loss(linear_model, residual_model, x_data_seq, u_data_seq)
        total_loss += loss.item()
    return total_loss / len(data_loader)



def train_residual_model(linear_model, residual_model, dataset, config, learning_rate = 0.001, device = 'cpu'):

    if linear_model is not None:
        linear_model.to(device)
        linear_model.eval()
    
    if residual_model is None:
        raise ValueError("The residual model must be provided.")
    
    residual_model.to(device)

    # Load the dataset
    x_data, _, u_data = dataset

    x_data_seq = cut_slices(x_data, config['window_size'] - 1, config['predict_num'])
    u_data_seq = cut_slices(u_data, config['window_size'] - 1, config['predict_num'])

    # Split the data into training and validation sets
    x_train, x_val, u_train, u_val = train_test_split(x_data_seq, u_data_seq, test_size = 0.3, random_state = 42)

    # Create a DataLoader for the training data and validation data
    train_dataset = torch.utils.data.TensorDataset(x_train, u_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, u_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = False)

    # Define the optimizer
    optimizer = optim.Adam(residual_model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.9)

    # Define the loss history
    loss_history = []
    val_loss_history = []

    # Train the model
    best_loss = float('inf')
    best_model_state = None
    for epoch in range(config['num_epochs']):
        loss = train_one_epoch(linear_model, residual_model, optimizer, train_loader, epoch)
        scheduler.step()
        loss_history.append(loss)
        val_loss = test_one_epoch(linear_model, residual_model, val_loader)
        val_loss_history.append(val_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(residual_model.state_dict())

    # Load the best model state
    if best_model_state is not None:
        residual_model.load_state_dict(best_model_state)
    else:
        print("Warning: Best model state is None.")
    
    return residual_model, loss_history, val_loss_history

def iterative_training(dataset, linear_model, residual_model, config, learning_rate = 0.001, device = 'cpu'):
    # Initialize the linear model
    x_data, y_data, u_data = dataset
    linear_model = linear_regression((x_data, y_data, u_data), linear_model, device)

    # Iterative Training
    num_iter = config['num_iter']
    iterative_losses = []
    for iter in range(num_iter):
        print(f"Iteration {iter + 1}")
        residual_model, loss_history, val_loss_history = train_residual_model(linear_model, residual_model, dataset, config, learning_rate, device)
        linear_model = train_linear_model(dataset, linear_model, residual_model, device)
        np.save(f"residual_model_{iter + 1}_loss_history.npy", np.array(loss_history))
        np.save(f"residual_model_{iter + 1}_val_loss_history.npy", np.array(val_loss_history))   

        linear_predict = linear_model(x_data, u_data)
        residual_predict = residual_model(x_data, u_data)
        y_predict = linear_predict + residual_predict

        mse_loss = nn.MSELoss()
        loss = mse_loss(y_predict, y_data)
        print(f'Iterative Loss at ({iter+1}): {loss.item()}')
        iterative_losses.append(loss.item())
    return linear_model, residual_model, iterative_losses

def main():
    # Parse command-line arguments
    args = parse_arguments()
    config = read_config_file(args.config)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_dir = config['train_data_dir']
    x_data, y_data, u_data = data_preparation(config, data_dir)
    x_data = torch.tensor(x_data, dtype = torch.float64)
    y_data = torch.tensor(y_data, dtype = torch.float64)
    u_data = torch.tensor(u_data, dtype = torch.float64)

    # rescale and reduce dimension
    x_dim = x_data.shape[1]
    u_dim = u_data.shape[1]

    # Standardize the data
    x_mean_1 = torch.mean(x_data, dim = 0)
    x_std_1 = torch.std(x_data, dim = 0)
    std_layer_1 = StdScalerLayer(x_mean_1, x_std_1)
    x_data_scaled = std_layer_1.transform(x_data)

    u_mean = torch.mean(u_data, dim = 0)
    u_std = torch.std(u_data, dim = 0)
    std_layer_u = StdScalerLayer(u_mean, u_std)
    u_data_scaled = std_layer_u.transform(u_data)

    # Apply PCA to the data
    pca = PCA(n_components = config['pca_dim'])
    pca.fit(x_data_scaled.detach().cpu().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype = torch.float64).to(device)
    pca_layer = PCALayer(x_dim, config['pca_dim'], pca_matrix)

    # Standardize the data 2
    x_pca = pca_layer.transform(x_data_scaled)
    x_mean_2 = torch.mean(x_pca, dim = 0)
    x_std_2 = torch.std(x_pca, dim = 0)
    std_layer_2 = StdScalerLayer(x_mean_2, x_std_2)

    # Build StdPCA Layer
    rescale_pca_layer = Rescale_pca_layer(std_layer_1, std_layer_2, std_layer_u, pca_layer)

    # Build dataset
    x_data = rescale_pca_layer.transform_x(x_data)
    y_data = rescale_pca_layer.transform_x(y_data)
    u_data = rescale_pca_layer.transform_u(u_data)

    dataset = (x_data, y_data, u_data)

    # Define the linear model
    state_dim = config['pca_dim'] + 1
    control_dim = u_dim
    linear_model = Linear_model(state_dim, control_dim).to(device)

    # Define the pretrained model
    

