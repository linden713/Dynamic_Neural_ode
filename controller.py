import torch
from env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]

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
    error = state - target_pose
    q = torch.diag(torch.tensor([1, 1, 0.1]))
    cost = torch.sum(error @ q @ error.T,dim =1)
    # ---
    return cost


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    x = state[:, 0]
    y = state[:, 1]
    theta = state[:, 2]
    centers = torch.stack([x, y], dim=1) 
    r = box_size / 2.0
    B = state.shape[0]
    A1 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  
    A2 = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=1)  
    u1 = A1  
    u2 = A2
    u3 = torch.tensor([1.0, 0.0], device=state.device).expand(B, 2)
    u4 = torch.tensor([0.0, 1.0], device=state.device).expand(B, 2)
    
    def test_axis(u):
        dist = torch.abs(torch.sum((centers - obstacle_centre) * u, dim=1))
        proj_robot = r * (torch.abs(torch.sum(u * A1, dim=1)) + torch.abs(torch.sum(u * A2, dim=1)))
        proj_obs = (obstacle_dims[0] / 2.0) * torch.abs(u[:, 0]) + (obstacle_dims[1] / 2.0) * torch.abs(u[:, 1])
        return dist <= (proj_robot + proj_obs)
    
    test1 = test_axis(u1)
    test2 = test_axis(u2)
    test3 = test_axis(u3)
    test4 = test_axis(u4)

    collision = test1 & test2 & test3 & test4
    in_collision = collision.to(torch.float)
    # ---
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    error = state - target_pose
    q = torch.diag(torch.tensor([1, 1, 0.1]))
    cost = torch.sum(error @ q @ error.T,dim =1) + 100 * collision_detection(state)
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
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.4 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
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
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.tensor(state, dtype=torch.float32)  
        action_tensor = self.mppi.command(state_tensor)
        action = action_tensor.detach().cpu().numpy()  
        # ---
        return action