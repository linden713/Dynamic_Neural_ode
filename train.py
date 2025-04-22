import torch
import numpy as np
import matplotlib.pyplot as plt

from model.losses import MultiStepLoss, SE2PoseLoss
from datasets import process_data_multiple_step
from tqdm import tqdm
import torch.optim as optim
from model.nn_dynamics_models import ResidualDynamicsModel
from env.panda_pushing_env import PandaPushingEnv


def train_step(model, train_loader, optimizer) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0. # TODO: Modify the value
    # Initialize the train loop
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        state = batch['state']
        action = batch['action']
        next_state_gth = batch['next_state']
        optimizer.zero_grad()
        loss = pose_loss(model, state, action, next_state_gth)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)


def val_step(model, val_loader) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0. # TODO: Modify the value
    # Initialize the validation loop
    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        loss = None
        state = batch['state']
        action = batch['action']
        next_state_gth = batch['next_state']
        loss = pose_loss(model, state, action, next_state_gth)
        val_loss += loss.item()
    return val_loss/len(val_loader)


def train_model(model, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    optimizer = None
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = None
        val_loss_i = None
        train_loss_i = train_step(model, train_dataloader, optimizer)
        val_loss_i = val_step(model, val_dataloader)
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    return train_losses, val_losses


pushing_multistep_residual_dynamics_model = None
collected_data = np.load("collected_data.npy", allow_pickle=True)  # 改成你的路径

train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=500)

pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
pose_loss = MultiStepLoss(pose_loss, discount=0.9)

env = PandaPushingEnv()
env.reset()

# --- Your code here
pushing_multistep_residual_dynamics_model = ResidualDynamicsModel(env.observation_space.shape[0], env.action_space.shape[0])

LR = 0.0001
NUM_EPOCHS = 5000

train_losses, val_losses = train_model(pushing_multistep_residual_dynamics_model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LR)


# plot train loss and test loss:
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
axes[0].plot(train_losses)
axes[0].grid()
axes[0].set_title('Train Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Train Loss')
axes[0].set_yscale('log')
axes[1].plot(val_losses)
axes[1].grid()
axes[1].set_title('Validation Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Validation Loss')
axes[1].set_yscale('log')

# ---

# save model:
torch.save(pushing_multistep_residual_dynamics_model.state_dict(), 'checkpoint/pushing_multi_step_residual_dynamics_model.pt')