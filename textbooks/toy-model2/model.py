import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class diffusion_equation(object):
    def __init__(self, dim = 20, L0 = 0, L1 = 1, mu = [1, 1]):
        self.dim = dim
        self.L0 = L0
        self.L1 = L1
        self.mu = mu
        self.build_A()

    def build_A(self):
        self.A = np.zeros([self.dim - 1, self.dim - 1])
        for i in range(self.dim - 1):
            self.A[i, i] = -2 
        for i in range(self.dim - 2):
            self.A[i, i + 1] = 1
            self.A[i + 1, i] = 1
    
    def generate_traj(self, steps, u0, dlt_t = 0.001):
        u_data = [u0]
        for _ in range(steps):
            u0 = u0 + dlt_t * (self.mu[0] * self.A @ u0 + self.mu[1] * u0 @ self.A)
            u_data.append(u0)
        return np.array(u_data)
    
class diffusion_equation(object):
    def __init__(self, dim = 20, L0 = 0, L1 = 1, mu = [1, 1]):
        self.dim = dim
        self.L0 = L0
        self.L1 = L1
        self.mu = mu
        self.build_A()

    def build_A(self):
        self.A = np.zeros([self.dim - 1, self.dim - 1])
        for i in range(self.dim - 1):
            self.A[i, i] = -2 
        for i in range(self.dim - 2):
            self.A[i, i + 1] = 1
            self.A[i + 1, i] = 1
    
    def generate_traj(self, steps, u0, dlt_t = 0.001):
        u_data = [u0]
        for _ in range(steps):
            u0 = u0 + dlt_t * (self.mu[0] * self.A @ u0 + self.mu[1] * u0 @ self.A)
            u_data.append(u0)
        return np.array(u_data)
    
    def generate_training_data(self, traj_num, steps, dlt_t = 0.001):
        u0 = np.random.rand(self.dim - 1, self.dim - 1)
        u_data = self.generate_traj(steps, u0, dlt_t)
        u_x = u_data[:-1,:,:]
        u_y = u_data[1:, :, :]
        for i in range(traj_num - 1):
            u0 = np.random.rand(self.dim - 1, self.dim - 1)
            u = self.generate_traj(steps, u0, dlt_t)
            u_data = np.concatenate((u_data, u), axis = 0)
            u_x = np.concatenate((u_x, u[:-1,:,:]), axis = 0)
            u_y = np.concatenate((u_y, u[1:,:,:]), axis = 0)
        u_x_lace_1 = u_x @ self.A
        u_x_lace_2 = self.A @ u_x

        u_x_tensor = torch.tensor(u_x, dtype=torch.float32)
        u_y_tensor = torch.tensor(u_y, dtype=torch.float32)
        u_x_lace_1_tensor = torch.tensor(u_x_lace_1, dtype=torch.float32)
        u_x_lace_2_tensor = torch.tensor(u_x_lace_2, dtype=torch.float32)
        u_x_tensor = u_x_tensor.reshape((-1, 1))
        u_y_tensor = u_y_tensor.reshape((-1, 1))
        u_x_lace_1_tensor = u_x_lace_1_tensor.reshape((-1, 1))
        u_x_lace_2_tensor = u_x_lace_2_tensor.reshape((-1, 1))

        dataset = TensorDataset(u_x_tensor, u_y_tensor, u_x_lace_1_tensor, u_x_lace_2_tensor)
    
        return dataset

    
class reaction_diffusion_equation(object):
    def __init__(self, dim = 20, L0 = 0, L1 = 1, mu = 1):
        self.dim = dim
        self.L0 = L0
        self.L1 = L1
        self.mu = mu
        self.build_A()
    
    def reaction_term(self, u):
        return 100 * 1/4*2 * (u-1)*(u+1)**2+ 1/4 * 2 * (u-1)**2 * (u+1)
    
    def build_A(self):
        self.A = np.zeros([self.dim - 1, self.dim - 1])
        for i in range(self.dim - 1):
            self.A[i, i] = -2 
        for i in range(self.dim - 2):
            self.A[i, i + 1] = 1
            self.A[i + 1, i] = 1
    
    def generate_traj(self, steps, u0, dlt_t = 0.001):
        u_data = np.zeros([steps + 1, self.dim - 1])
        u_data[0, :] = u0
        dlt_x = (self.L1 - self.L0)/self.dim
        for step in range(steps):
            u_0 = u_data[step, :]
            u_1 = u_0 + dlt_t * (self.mu * u_0 @ self.A/ dlt_x ** 2 + self.reaction_term(u_0))
            u_data[step + 1,:] = u_1
        return u_data

    def generate_training_data(self, steps, traj_num, dlt_t = 0.001):
        u0 = 2 * np.random.rand(self.dim - 1) - 1

        u_data = self.generate_traj(steps, u0, dlt_t)

        u_x = u_data[:-1,:]
        u_y = u_data[1:,:]
        for _ in range(traj_num - 1):
            u0 = np.random.rand(self.dim - 1)
            u = self.generate_traj(steps, u0, dlt_t)
            u_data = np.concatenate((u_data, u), axis = 0)
            u_x = np.concatenate((u_x, u[:-1,:]), axis = 0)
            u_y = np.concatenate((u_y, u[1:,:]), axis = 0)
        u_x_lace = u_x @ self.A

        u_x = u_x.reshape((-1, 1))
        u_y = u_y.reshape((-1, 1))
        u_x_lace = u_x_lace.reshape((-1, 1))

        u_x_tensor = torch.tensor(u_x, dtype=torch.float32)
        u_y_tensor = torch.tensor(u_y, dtype=torch.float32)
        u_x_lace_tensor = torch.tensor(u_x_lace, dtype=torch.float32)

        dataset = TensorDataset(u_x_tensor, u_y_tensor, u_x_lace_tensor)

        return dataset
