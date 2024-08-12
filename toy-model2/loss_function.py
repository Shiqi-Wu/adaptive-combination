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


def terminal_loss_ver2(g, h, y):
    gh = torch.cat((g, h), 1)

    # Calculate projection coefficients
    gh_orthogonal = gram_schmidt(gh)
    coeffs, _, _, _ = torch.linalg.lstsq(gh_orthogonal, y)
    # print(coeffs)

    # Calculate the projection of y onto the space spanned by g and h
    y_proj = torch.matmul(gh_orthogonal, coeffs)

    # Calculate loss (MSE)
    loss = F.mse_loss(y_proj, y)
    return loss

def gram_schmidt(vectors, epsilon=1e-10):
    orthogonal_vectors = []
    for i in range(vectors.shape[1]):
        v = vectors[:, i]
        for u in orthogonal_vectors:
            v = v - torch.dot(v, u) * u  # Subtract the projection of v onto u

        # Check if the vector is close to zero before normalizing
        norm_v = torch.norm(v)
        if norm_v > epsilon:  # Only normalize if the norm is greater than a small epsilon
            orthogonal_vectors.append(v / norm_v)
        else:
            orthogonal_vectors.append(v)  # If norm is too small, avoid division

    return torch.stack(orthogonal_vectors, dim=1)

def orthogonal_loss_ver2(g, h):
    
    # Orthogonalize g and h separately
    g_orthogonal = gram_schmidt(g)
    h_orthogonal = gram_schmidt(h)

    # Calculate cosine similarity
    cosine_sim = torch.sum((g_orthogonal * h_orthogonal)**2)
    return cosine_sim