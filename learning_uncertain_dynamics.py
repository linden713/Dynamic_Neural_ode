import torch

from torch.distributions import MultivariateNormal
from env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, DISK_RADIUS, OBSTACLE_RADIUS, OBSTACLE_CENTRE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)[:2]
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)[:2]



def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    # Compute the distance to the target pose
    distance = torch.norm(state[..., :2] - target_pose, dim=-1)  # shape (B,)
    diagonal = torch.diagonal(state[...,2:].reshape(state.shape[0],2,2),dim1=-2,dim2=-1)
    cost = distance ** 2  + diagonal.sum(dim=-1)
    
    # ---
    return cost


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.

    :param state: torch tensor of shape (B, dx + dx^2) First dx consists of state mean,
                  the rest is the state covariance matrix
    :param action: torch tensor of shape (B, du)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    obs_centre = torch.from_numpy(OBSTACLE_CENTRE).to(dtype=torch.float32)
    obs_radius = OBSTACLE_RADIUS
    disk_radius = DISK_RADIUS
    cost = None
    # --- Your code here

    collision = (obs_radius + disk_radius) > torch.norm(state[...,:2] - obs_centre, dim=-1)  # shape (B,)
    
    distance = torch.norm(state[..., :2] - target_pose, dim=-1)  # shape (B,)
    diagonal = torch.diagonal(state[...,2:].reshape(state.shape[0],2,2),dim1=-2,dim2=-1)
    cost = distance ** 2  + diagonal.sum(dim=-1) + collision * 100
    # ---
    return cost


def obstacle_avoidance_pushing_cost_function_samples(state, action, K=10):
    """
    Compute the state cost for MPPI on a setup with obstacles, using samples to evaluate the expected collision cost

    :param state: torch tensor of shape (B, dx + dx^2) First dx consists of state mean,
                  the rest is the state covariance matrix
    :param action: torch tensor of shape (B, du)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    obs_centre = torch.from_numpy(OBSTACLE_CENTRE).to(dtype=torch.float32)
    obs_radius = OBSTACLE_RADIUS
    disk_radius = DISK_RADIUS
    cost = None
    # --- Your code here
    state_distribution = MultivariateNormal(state[..., :2], covariance_matrix=state[..., 2:].reshape(state.shape[0], 2, 2))  # shape (B, dx)
    state_samples = state_distribution.sample((K,))
    collision_diatance = torch.norm(state_samples - obs_centre, dim=-1)
    collision = (obs_radius + disk_radius) > collision_diatance.float()
    # print(collision.shape)
    distance = torch.norm(state[..., :2] - target_pose, dim=-1)  # shape (B,)
    diagonal = torch.diagonal(state[...,2:].reshape(state.shape[0],2,2),dim1=-2,dim2=-1)
    cost = distance ** 2  + diagonal.sum(dim=-1) + collision.float().mean(dim=0) * 100
    # ---
    return cost


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low) + 1e-5
        u_max = torch.from_numpy(env.action_space.high) - 1e-5
        noise_sigma = 0.25 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.001
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim + state_dim * state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size + state_size**2)
                      consisting of state mean and flattened covariance
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size + state_size**2) containing the predicted states mean
                 and covariance
        """
        next_state = None
        # --- Your code here 
        if True:
        # Unpack the state tensor into mean and covariance
            state_mean = state[..., :2]  # shape (B, 2)
            state_cov = state[...,2:].reshape(state.shape[0],2,2)  # shape (B, 2, 2)
            # Use the model to predict the next state mean and covariance
            next_state_mean, next_state_cov = self.model.propagate_uncertainty(state_mean, state_cov, action)
            # Concatenate the next state mean and flattened covariance
            next_state = torch.cat((next_state_mean, next_state_cov.flatten(start_dim=1)), dim=-1)  # shape (B, 2 + 2*2)
        else:
            next_state = self.model(state, action)

        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Recall that our current state is (state_size,), but we have set up our dynamics to work with state means and
           covariances. You need to use the current state to initialize a (mu, sigma) tensor. Given that we know the
           current state, the initial sigma should be zero.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        if True:
            # Prepare the state tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # shape (1, state_size)
            # Initialize the state mean and covariance
            state_tensor = torch.cat((state_tensor, torch.zeros(1, 2*2)), dim=-1)  # shape (1, state_size + state_size**2)
            # ---
            action_tensor = self.mppi.command(state_tensor)
            # --- Your code here
            action = action_tensor.squeeze(0).detach().numpy()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)  # 转换为 (1, state_size) 形状的 Torch Tensor
            # print(state_tensor.shape)
            # ---
            action_tensor = self.mppi.command(state_tensor)
            # --- Your code here
            action = action_tensor.detach().cpu().numpy()  # 转换回 numpy 数组，确保形状为 (action_size,
            # ---
        return action
