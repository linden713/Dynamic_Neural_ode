import torch
import torch.nn as nn
import torch.nn.functional as F


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        x_pred, y_pred, theta_pred = pose_pred[..., 0], pose_pred[..., 1], pose_pred[..., 2]
        x_target, y_target, theta_target = pose_target[..., 0], pose_target[..., 1], pose_target[..., 2]
        
        device = pose_pred.device # algin device 
        w_tensor = torch.tensor(self.w, device=device)
        l_tensor = torch.tensor(self.l, device=device)
        rg = torch.sqrt((l_tensor**2 + w_tensor**2) / 12)
        
        se2_pose_loss = F.mse_loss(x_pred, x_target, reduction='mean') +\
                        F.mse_loss(y_pred, y_target, reduction='mean') +\
                        rg * F.mse_loss(theta_pred, theta_target, reduction='mean')
        # ---
        return se2_pose_loss


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here
        pred_state = model(state, action)
        single_step_loss = self.loss(pred_state, target_state)
        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        batch_size, num_steps = actions.shape[:2]
        discounts = (self.discount ** torch.arange(num_steps))
        pred_states = torch.empty((batch_size, num_steps+1, model.state_dim))
        pred_states[:, 0] = state  
        for t in range(num_steps):
            current_state = pred_states[:, t]
            pred_states[:, t+1] = model(current_state, actions[:, t])
        errors = self.loss(pred_states[:, 1:], target_states) 
        weighted_errors = errors * discounts[None, :]  
        multi_step_loss = weighted_errors.sum()

        return multi_step_loss
