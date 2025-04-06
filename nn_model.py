from torch import nn
import torch 
from torch.distributions import MultivariateNormal

class PushingDynamics(nn.Module):
    def __init__(self, propagation_method):
        super().__init__()
        self.propagation_method = propagation_method
        
    def forward(self, state, action):
        raise NotImplementedError

    def predict(self, state, action):
        raise NotImplementedError

    def propagate_uncertainty(self, mu, sigma, action):
        if self.propagation_method == 'certainty_equivalence':
            return self.propagate_uncertainty_certainty_equivalence(mu, sigma, action)
        if self.propagation_method == 'linearization':
            return self.propagate_uncertainty_linearization(mu, sigma, action)
        if self.propagation_method == 'moment_matching':
            return self.propagate_uncertainty_moment_matching(mu, sigma, action)

        raise ValueError('invalid self.propagation_method')

    def propagate_uncertainty_linearization(self, mu, sigma, action):
        raise NotImplementedError

    def propagate_uncertainty_moment_matching(self, mu, sigma, action, K=50):
        """
        Propagate uncertainty via moment matching with samples
        Args:
            mu: torch.tensor of shape (N, dx) consisting of mean of current state distribution
            sigma: torch.tensor of shape (N, dx, dx) covariance matrix of current state distribution
            action: torch.tensor of shape (N, du) action

        Returns:
            pred_mu: torch.tensor of shape (N, dx) consisting of mean of predicted state distribution
            pred_sigma: torch.tensor of shape (N, dx, dx) consisting of covariance matrix of predicted state distribution

        """

        pred_mu, pred_sigma = None, None
        # print(f"mu:{mu}")
        # print(f"sigma:{sigma}")
        # print(f"action:{action}")
        import os
        import datetime


        # --- Your code here
        # if ensembles = 1
        if self.num_ensembles == 1:
            return self.predict(mu, action) 

        
        if torch.all(sigma == 0):
            pred_mu,pred_sigma = self.predict(mu, action)  # shape (N, dx)
        else:

            state_distribution = MultivariateNormal(mu, covariance_matrix=sigma)  # shape (N, dx)
            samples = state_distribution.sample((K,))  # shape (K, N, dx)

            sample_mu =[]
            for k in range(K):
                next_state_mu,next_state_sigma = self.predict(samples[k,:], action)  
                next_state_distribution = MultivariateNormal(next_state_mu, covariance_matrix=next_state_sigma)  # shape (K, N, dx)
                sample_mu.append(next_state_distribution.sample())  # shape (N, dx)
            pred_mu=torch.stack(sample_mu,dim=0).mean(dim=0)  # shape (N, dx)
            pred_sigma = batch_cov(torch.stack(sample_mu,dim=0).permute(1, 0, 2))  # shape (N, dx, dx)
        # ---


        return pred_mu, pred_sigma
class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.linear1 = nn.Linear(state_dim + action_dim,100)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(100,100)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(100, state_dim)
        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        x = torch.cat((state,action),dim =-1)
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        next_state = self.linear3(x) + state
        # ---
        return next_state


class DynamicsNNEnsemble(PushingDynamics):

    def __init__(self, state_dim, action_dim, num_ensembles, propagation_method='moment_matching'):
        assert propagation_method in ['moment_matching', 'certainty_equivalence']
        super().__init__(propagation_method)
        self.models = nn.ModuleList([ResidualDynamicsModel(state_dim, action_dim) for _ in range(num_ensembles)])
        self.propagation_method = propagation_method
        self.num_ensembles = num_ensembles

    def forward(self, state, action):
        """
            Forward function for dynamics ensemble
            You should use this during training
        Args:
            state: torch.tensor of shape (B, dx)
            action: torch.tensor of shape (B, du)

        Returns:
            Predicted next state for each of the ensembles
            next_state: torch.tensor of shape (B, N, dx) where N is the number of models in the ensemble

        """
        next_state = None
        # --- Your code here
        next_state = torch.stack([model(state, action) for model in self.models], dim=1)  # shape (B, N, dx)
        # ---
        return next_state

    def predict(self, state, action):
        """
            Predict function for NN ensemble
            You should use this during evaluation
            This will return the mean and covariance of the ensemble output
         Args:
            state: torch.tensor of shape (B, dx)
            action: torch.tensor of shape (B, du)

        Returns:
            Predicted next state for each of the ensembles
            pred_mu : torch.tensor of shape (B, dx)
            pred_sigma: torch.tensor of shape (B, dx, dx) covariance matrix

        """
        pred_mu, pred_sigma = None, None
        # --- Your code here
        next_state = self.forward(state,action) 
        pred_mu = next_state.mean(dim=1)
        assert next_state.dim() == 3, f"Expected shape (B, N, D), got {next_state.shape}"

        pred_sigma = batch_cov(next_state)  # shape (B, dx, dx)
        # ---
        return pred_mu, pred_sigma
    
    



def batch_cov(points):
    """
    Estimates covariance matrix of batches of sets of points
    Args:
        points: torch.tensor of shape (B, N, D), where B is the batch size, N is sample size,
                and D is the dimensionality of the data

    Returns:
        bcov: torch.tensor of shape (B, D, D) of B DxD covariance matrices
    """
    B, N, D = points.size()
    if N == 1:
        return torch.zeros(B, D, D, device=points.device)
    
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)