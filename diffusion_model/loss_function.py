import torch


# Define a function to orthogonalize the columns of a matrix
def orthogonalize_columns(matrix):
    """
    Orthogonalize the columns of the input matrix using QR decomposition.

    Parameters:
    matrix (torch.Tensor): The input matrix of shape (batch_size, n_features)

    Returns:
    torch.Tensor: The orthogonalized matrix with the same shape as the input.
    """
    # Use QR decomposition to obtain an orthogonal matrix
    q, _ = torch.linalg.qr(matrix)
    return q

def loss_function_orth(G, H):
    """
    Compute the loss while preserving gradient computation for H.

    Parameters:
    G (torch.Tensor): Input matrix G of shape (d, n), orthogonalized columns.
    H (torch.Tensor): Input matrix H of shape (m, d), requires gradient.

    Returns:
    torch.Tensor: The computed loss value.
    """
    # Step 1: Compute H @ H^T and its pseudoinverse
    H_TH = H.T @ H
    H_TH_pinv = torch.linalg.pinv(H_TH)

    # Step 2: Compute the projection matrix: H^T * (H H^T)^dagger * H
    projection_matrix = H @ H_TH_pinv @ H.T

    # Step 3: Compute the maximum loss over all column vectors of G
    max_loss = torch.tensor(float('-inf'), device=H.device, dtype=H.dtype)
    for k in range(G.shape[1]):
        g_k = G[:, k]  # Extract the k-th column vector of G

        # Ensure `g_k` is treated as a column vector for matrix multiplication
        g_k = g_k.view(-1, 1)  # Shape (d, 1)

        # Compute the loss term: g_k.T @ projection_matrix @ g_k
        term = (g_k.T @ projection_matrix @ g_k).squeeze()  # Shape () (scalar with gradient)

        # Use `torch.max` to preserve gradient flow
        max_loss = torch.max(max_loss, term)

    return max_loss


