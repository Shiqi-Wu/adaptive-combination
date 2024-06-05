import numpy as np
import torch
import torch.nn.functional as F



def terminal_loss(gh_outputs_orthogonal, y):
    d_dim = gh_outputs_orthogonal.shape[1]
    proj_coefficient_values = torch.zeros(d_dim, device=gh_outputs_orthogonal.device)
    
    # Calculate projection coefficients
    for i in range(d_dim):
        gh_i = gh_outputs_orthogonal[:, i]
        proj_coefficient = torch.dot(y.squeeze(1), gh_i) / torch.dot(gh_i, gh_i)
        proj_coefficient_values[i] = proj_coefficient
        # print(proj_coefficient)
    
    # Calculate projected y
    proj_y = torch.matmul(gh_outputs_orthogonal, proj_coefficient_values.unsqueeze(1)).squeeze()
    
    # Calculate loss (MSE)
    loss = F.mse_loss(proj_y, y.squeeze())
    
    return loss

def orthogonal_loss(g_outputs, h_outputs_orthogonal):
    g_dim = np.shape(g_outputs)[1]
    h_dim = np.shape(h_outputs_orthogonal)[1]
    proj_g_norm_squared = 0
    for i in range(g_dim):
        g_i = g_outputs[:, i]
        for j in range(h_dim):
            h_i = h_outputs_orthogonal[:, i]
            proj_coefficient = torch.dot(g_i.squeeze(), h_i) / torch.dot(h_i, h_i)
            proj_g_norm_squared += proj_coefficient ** 2
    
    loss = proj_g_norm_squared
    return loss
