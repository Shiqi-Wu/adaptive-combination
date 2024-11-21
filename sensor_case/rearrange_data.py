import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "utils"))

def load_dataset(data_dict, predict_num = 1):
    time = []
    data = []
    I_p = []
    for _, contents in data_dict.items():
        time.append(contents['time'])
        data.append(contents['data'])
        I_p.append(contents['I_p'])
    
    data = np.array(data)
    x_data = data[:-predict_num,:]
    y_data = data[predict_num:,:]
    # I_p = np.reshape(np.array(I_p)[:-1], (-1,1))
    u1_data = np.concatenate((np.reshape(np.array(I_p)[:-predict_num], (-1,1)), np.reshape(np.array(I_p)[1:data.shape[0]-predict_num+1], (-1,1))), axis = 1)
    u2_data = np.concatenate((np.reshape(np.array(I_p)[predict_num:], (-1,1)), np.reshape(np.concatenate((np.array(I_p)[predict_num+1:], np.array([I_p[0]]))), (-1,1))), axis = 1)
    return x_data, y_data, u1_data, u2_data

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

    return x_data, y_data, u_data

def cut_slices(data, window_size, predict_num):
    """
    Inputs:
        data (torch.Tensor): Input data of shape (num_samples, num_features).
        window_size (int): The size of the sliding window.
        predict_num (int): The number of consecutive steps to predict in each slice.
    Function:
        Extracts overlapping slices from the input data using a sliding window approach.
    Returns:
        data_slices (torch.Tensor): Extracted slices of shape (num_slices, predict_num, num_features).
    """
    if data.dim() != 2:
        raise ValueError("Input data must be a 2D tensor of shape (num_samples, num_features).")

    num_samples, num_features = data.shape
    if window_size < predict_num:
        raise ValueError("Window size must be greater than or equal to predict_num.")

    # Generate indices for slices
    slices = []
    for i in range(0, num_samples, window_size):
        if i + window_size <= num_samples:
            window = data[i:i + window_size]
            for j in range(window_size - predict_num + 1):
                slices.append(window[j:j + predict_num])

    # Stack slices into a single tensor
    data_slices = torch.stack(slices, dim=0)
    return data_slices
