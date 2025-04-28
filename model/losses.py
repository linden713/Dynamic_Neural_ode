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

    def __init__(self, loss_fn, discount=0.99, ):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        batch_size, num_steps = actions.shape[:2]
        device = state.device # Get device from input state

        # Ensure discounts are on the correct device
        discounts = (self.discount ** torch.arange(num_steps).to(device))

        multi_step_loss = 0.0
        current_state = state # Initial state

        # Rollout loop
        for t in range(num_steps):
            # Predict next state using the model
            pred_next_state = model(current_state, actions[:, t])

            # Calculate loss for the current step prediction vs target
            # Assumes target_states has shape (B, num_steps, StateDim)
            step_loss = self.loss(pred_next_state, target_states[:, t])

            # Apply discount and accumulate
            multi_step_loss += discounts[t] * step_loss

            # Update current_state for the next iteration
            # Use detach() if you DON'T want gradients flowing through predicted states
            # across time steps (often desired in model-based RL rollouts,
            # but maybe not for pure system ID depending on goal).
            # Keep as is if you DO want gradients flowing through the whole sequence.
            current_state = pred_next_state

        # Return the total discounted loss (implicitly averaged by step_loss if loss_fn averages)
        # Or divide by num_steps for average discounted loss per step
        return multi_step_loss # / num_steps
