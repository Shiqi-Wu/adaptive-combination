import numpy as np
import torch
import torch.nn.functional as F



def terminal_loss(gh_outputs_orthogonal, y):
    epsilon = 1e-6
    d_dim = gh_outputs_orthogonal.shape[1]
    proj_coefficient_values = torch.zeros(d_dim, device=gh_outputs_orthogonal.device)
    
    # Calculate projection coefficients
    for i in range(d_dim):
        gh_i = gh_outputs_orthogonal[:, i]
        proj_coefficient = torch.dot(y.squeeze(1), gh_i)
        proj_coefficient_values[i] = proj_coefficient
        # print(proj_coefficient)
    
    # Calculate projected y
    proj_y = torch.matmul(gh_outputs_orthogonal, proj_coefficient_values.unsqueeze(1)).squeeze()
    
    # Calculate loss (MSE)
    loss = F.mse_loss(proj_y, y.squeeze())
    
    return loss

def orthogonal_loss(g_outputs, h_outputs_orthogonal):
    epsilon = 1e-4
    smooth_param = 1e-3
    g_dim = np.shape(g_outputs)[1]
    h_dim = np.shape(h_outputs_orthogonal)[1]
    proj_g_norm_squared = 0
    N = g_outputs.shape[0]
    for i in range(g_dim):
        g_i = g_outputs[:, i]
        for j in range(h_dim):
            h_j = h_outputs_orthogonal[:, j]
            # print(torch.dot(g_i, g_i))
            proj_coefficient = torch.dot(g_i.squeeze(), h_j.squeeze()) / (torch.sqrt(torch.dot(g_i.squeeze(), g_i.squeeze())) + epsilon)
            # print(torch.dot(g_i.squeeze(), h_j.squeeze()))
            # print(proj_coefficient)
            proj_g_norm_squared += proj_coefficient ** 2

    # smooth_term = torch.log(model.k ** 2 + 1)
    loss = proj_g_norm_squared 
    # + smooth_param * smooth_term
    return loss
