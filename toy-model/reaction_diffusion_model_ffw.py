import os
import numpy as np
import matplotlib.pyplot as plt
from toy_model import *
import sys 
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import argparse

import torch
import torch.nn as nn
import yaml


class ParaModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, m_dim, seed=None):
        super(ParaModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.relu_layers = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(hidden_dims[-1], m_dim)
        self.W = nn.Parameter(torch.randn(m_dim, m_dim), requires_grad=True)
    
    def forward(self, x):
        x = self.relu_layers(x)
        x = self.output_layer(x)
        return x

class LinearRegressionLayer(nn.Module):
    def __init__(self, input_size, output_size, seed = None):
        super(LinearRegressionLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias = False)
        if seed is not None:
            torch.manual_seed(seed)
        
    def forward(self, x):
        return self.linear(x)

class FeedforwardLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seed = None, nonlinearity = 'relu'):
        super(FeedforwardLayer, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            raise ValueError("Nonlinearity not recognized")
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.nonlinearity(x)
        x = self.linear2(x)
        return x

# def parse_args():
#     parser = argparse.ArgumentParser(description='Reaction-diffusion model')
#     parser.add_argument('--input_dim', type=int, default=2, help='input dimension')
#     parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 64], help='hidden dimensions')
#     parser.add_argument('--m_dim', type=int, default=16, help='output dimension')
#     parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
#     parser.add_argument('--batch_size', type=int, default=512, help='batch size')
#     parser.add_argument('--num_epoch', type=int, default=100, help='number of epochs')
#     parser.add_argument('--save_dir', type=str, default='reaction_diffusion_output/', help='directory to save model')
#     parser.add_argument('--save_name', type=str, help='model name')
#     parser.add_argument('--load_dir', type=str, default='reaction_diffusion_output/', help='directory to load model')
#     parser.add_argument('--load_name', type=str, default=None, help='model name')
#     parser.add_argument('--seed', type=int, default=0, help='random seed')
#     parser.add_argument('--pre_train', action='store_true', help='train model')
#     parser.add_argument('--nonlinearity', type = str, default = 'relu', help='nonlinearity')
#     args = parser.parse_args()
#     return args

def parse_args():
    parser = argparse.ArgumentParser(description='Reaction-diffusion model')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def custom_loss_function(h_outputs, y, lace, mu, lambda1, lambda2, m_dim):
    # Compute g as the product of mu and lace
    g = mu * lace
    # Compute u_x as the difference between y and g, and remove extra dimensions
    u_x = (y - g).squeeze()

    # Initialize the squared norm of g's projection onto the subspace spanned by h_outputs
    proj_g_norm_squared = 0

    # For each vector in the orthogonal basis h_outputs
    for i in range(m_dim):
        h_i = h_outputs[:, i]
        # Calculate the projection coefficient of g onto h_i and accumulate the squares of these projections
        proj_coefficient = (torch.dot(g.squeeze(), h_i) / torch.dot(h_i, h_i))
        proj_g_norm_squared += proj_coefficient ** 2

    # Initialize the squared norm of u_x's projection onto the subspace spanned by h_outputs
    proj_u_x_norm_squared = 0
    k_i_values = torch.zeros(m_dim, device=h_outputs.device)

    for i in range(m_dim):
        h_i = h_outputs[:, i]
        # Calculate the projection coefficient of u_x onto h_i
        proj_coefficient_u_x = (torch.dot(u_x, h_i) / torch.dot(h_i, h_i))
        proj_u_x_norm_squared += proj_coefficient_u_x ** 2
        k_i_values[i] = proj_coefficient_u_x

    # Define loss1 as the lambda1-weighted squared norm of g's projection
    loss1 = lambda1 * proj_g_norm_squared
    # Define loss2 based on the squared difference between u_x and its projection onto the subspace
    loss2 = lambda2 * torch.sum((torch.matmul(h_outputs, k_i_values.unsqueeze(1)).squeeze() - u_x) ** 2)

    # Calculate the total loss as the sum of loss1 and loss2
    total_loss = loss1 + loss2

    return total_loss, loss1, loss2

def pretrain_one_epoch(model, optim, data_loader, m_dim, epoch, mu, lambda1, lambda2):
    model.train()
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0

    print(f"Training Epoch: {epoch+1}")
    for x, y, lace in data_loader:
        
        h_outputs = model(torch.cat((x, lace), dim = 1))
        Q, R = torch.linalg.qr(h_outputs, mode='reduced')
        
        # R_inv = torch.linalg.inv(R)

        # with torch.no_grad():
            # model.W.data = R_inv

        # h_outputs = model.orthogonal(x)
        optim.zero_grad()
        loss, loss1, loss2 = custom_loss_function(Q, y, lace, mu, lambda1, lambda2, m_dim)
        loss.backward()
        optim.step()
        # print(loss.item())
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient of {name} is {param.grad.norm().item()}")  # 打印梯度的范数


        train_loss += loss.item()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
    

    avg_train_loss = train_loss / len(data_loader)
    avg_train_loss1 = train_loss1 / len(data_loader)
    avg_train_loss2 = train_loss2 / len(data_loader)
    print(f"Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4e}; Loss1: {avg_train_loss1:.4e}; Loss2: {avg_train_loss2:.4e}")

    return avg_train_loss, avg_train_loss1, avg_train_loss2


def optimize_g(lace, y):
    lace_mean = torch.mean(lace)
    y_mean = torch.mean(y)

    mu_numerator = torch.sum((lace - lace_mean) * (y - y_mean))
    mu_denominator = torch.sum((lace - lace_mean) ** 2)
    mu = mu_numerator / mu_denominator

    return mu

def pre_train(config):
    data_model = reaction_diffusion_equation()
    dataset = data_model.generate_training_data(500, 10, dlt_t=0.001)  # 注意这里的函数名可能需要修正为generate_training_data
    
    # Calculate mu
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) 
    for _, y, lace in dataloader:
        break

    mu = optimize_g(lace, y)

    # Initialize model
    model_h = ParaModel(config['input_dim'], config['hidden_dims'], config['m_dim'], seed=config['seed'])
    optimizer = Adam(model_h.parameters(), lr=config['lr'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    epochs = config['num_epoch']
    train_losses = [0] * epochs
    train_losses1 = [0] * epochs
    train_losses2 = [0] * epochs

    for epoch in range(epochs):
        train_loss, train_loss1, train_loss2 = pretrain_one_epoch(model_h, optimizer, dataloader, config['m_dim'], epoch, mu, config['lambda1'], config['lambda2'])
        train_losses[epoch] = train_loss
        train_losses1[epoch] = train_loss1
        train_losses2[epoch] = train_loss2
        scheduler.step()
    
    # Save model
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    torch.save(model_h, config['save_dir'] + config['save_name'] + '_pretrained.pth')
    print(f"Model saved to {config['save_dir'] + config['save_name']}_pretrained.pth")

    # Plot loss
    plt.plot(train_losses, label='Total Loss')
    plt.plot(train_losses1, label='Loss1')
    plt.plot(train_losses2, label='Loss2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(config['save_dir'] + config['save_name'] + '_pretrained_loss.png')
    plt.show()

def train_linear_layer(model, linear_layer, x, lace, y, config):
    for param in model.parameters():
        param.requires_grad = False
    
    optimizer = Adam(linear_layer.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()

    dataset = TensorDataset(x, y, lace)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    epochs = config.get('linear_layer_epochs', 300)  # Assuming 'linear_layer_epochs' is your desired key, with a default of 300 if not present
    
    for _ in range(epochs):
        model.eval()
        total_loss = 0

        for batch_x, batch_y, batch_lace in data_loader:
            inputs = torch.cat((batch_x, batch_lace), dim=1)
            outputs = model(inputs)
            predictions = linear_layer(outputs)

            loss = criterion(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        scheduler.step()

def iterative_train(dataset, model_h, linear_h, config):
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for x, y, lace in dataloader:
        break
    y_g = y

    iterations = config['iterations']
    err_history_h = [0] * iterations
    inputs = torch.cat((x, lace), dim=1)

    print('Starting iterative training')
    for iteration in range(iterations):
        mu_h = optimize_g(lace, y_g)
        
        y_h = y - mu_h * lace
        train_linear_layer(model_h, linear_h, x, lace, y_h, config)
        y_g = y - linear_h(model_h(inputs)).detach()
        err = torch.norm(y - y_h - y_g)
        print('iteration:', iteration, 'error:', err.item())
        err_history_h[iteration] = err.item()
    return err_history_h


if __name__ == "__main__":
    args = parse_args()
    
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError("No configuration file provided.")

    if config['pre_train']:
        pre_train(config)
    else:
        if config.get('load_name') is None:
            print("Random number seed:", config['seed'])
            np.random.seed(config['seed'])
            torch.manual_seed(config['seed'])
            model_h = ParaModel(config['input_dim'], config['hidden_dims'], config['m_dim'], seed=config['seed'])
        else:
            model_h = torch.load(config['load_dir'] + config['load_name'])
            print(f"Model loaded from {config['load_dir'] + config['load_name']}")
        if config.get('nonlinearity') is not None:
            linear_h = FeedforwardLayer(input_size = config['m_dim'], hidden_size = config['m_dim'], output_size= 1, seed=config['seed'], nonlinearity=config.get('nonlinearity'))
        else:
            linear_h = LinearRegressionLayer(config['m_dim'], 1, seed=config['seed'])

        data_model = reaction_diffusion_equation()
        dataset = data_model.generate_training_data(500, 10, dlt_t=0.001)
        err_history_h = iterative_train(dataset, model_h, linear_h, config)

        # Save model
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        torch.save(model_h, config['save_dir'] + config['save_name'] + '.pth')
        print(f"Model saved to {config['save_dir'] + config['save_name']}.pth")
        torch.save(linear_h, config['save_dir'] + config['save_name'] + '_linear' + '.pth')
        print(f"Model saved to {config['save_dir'] + config['save_name'] + '_linear'}.pth")

        # Plot error
        plt.plot(range(1, config['iterations']+1), err_history_h, label='Error') # Corrected
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.yscale('log')
        plt.legend()
        plt.savefig(config['save_dir'] + config['save_name'] + '_iterative_error.png')
        plt.show()

