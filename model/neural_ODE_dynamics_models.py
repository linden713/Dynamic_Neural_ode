import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim,100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.state_dim)
        )

    def forward(self, t, x_aug):
        state, action = x_aug[:, :self.state_dim], x_aug[:, self.state_dim:]
        dxdt = self.net(torch.cat([state, action], dim=-1))
        return torch.cat([dxdt, torch.zeros_like(action)], dim=-1)

class NeuralODEDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, ode_method='dopri5'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ode_func = ODEFunc(state_dim, action_dim)
        self.ode_method = ode_method
        
    def forward(self, state, action):
        # Concatenate state and action to form augmented input
        x_aug = torch.cat([state, action], dim=-1)  # (B, state+action)
        t = torch.tensor([0, 1], dtype=torch.float32).to(state.device)  # Integration time
        traj = odeint(self.ode_func, x_aug, t, method=self.ode_method)  # (2, B, state+action)
        return traj[1][:, :self.state_dim]  # Return state after integration
