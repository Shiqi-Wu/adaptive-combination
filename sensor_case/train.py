import numpy as np
from sklean.model_selection import train_test_split
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

# Read config file
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

def read_config_file(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def linear_regression(dataset, linear_model, device='cpu'):
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

### Load data
def data_preparation(config, data_dir):
    window_size = config['window_size']
    print(f'window_size: {window_size}')
    x_dataset, y_dataset, u_dataset = [], [], []
    # Load data
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)

        # Check if the file exists before trying to load it
        
        if os.path.exists(data_file_path) and data_file_path.endswith('.npy'):
            
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data, y_data, u_data, _ = load_dataset(data_dict)
            # print(x_data.shape, y_data.shape, u_data.shape)
            x_dataset.append(x_data[1:window_size])
            y_dataset.append(y_data[1:window_size])
            u_dataset.append(u_data[1:window_size])
            # print(f"Loaded data from {data_file_path}")
        else:
            print(f"File not found: {data_file_path}")

    # Concatenate data
    x_data = np.concatenate(x_dataset, axis=0)
    y_data = np.concatenate(y_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)
    print(f'x_data shape: {x_data.shape}, y_data shape: {y_data.shape}, u_data shape: {u_data.shape}')

    return x_data, y_data, u_data

