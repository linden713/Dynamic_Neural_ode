import torch
import numpy as np
import matplotlib.pyplot as plt

from model.losses import MultiStepLoss, SE2PoseLoss
from datasets import process_data_multiple_step
from tqdm import tqdm
import torch.optim as optim
from model.nn_dynamics_models import ResidualDynamicsModel
from env.panda_pushing_env import PandaPushingEnv
from model.neural_ODE_dynamics_models import NeuralODEDynamicsModel

# parameters
# MODEL = "residual_nn" 
MODEL = "ode"
LR = 0.0001
NUM_EPOCHS = 3000
NUM_STEPS = 4
BATCH_SIZE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
def train_step(model, train_loader, optimizer) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0. 
    # Initialize the train loop
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        state = batch['state'].to(DEVICE)
        action = batch['action'].to(DEVICE)
        next_state_gth = batch['next_state'].to(DEVICE)
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
    val_loss = 0. 
    # Initialize the validation loop
    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        loss = None
        state = batch['state'].to(DEVICE)
        action = batch['action'].to(DEVICE)
        next_state_gth = batch['next_state'].to(DEVICE)
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


if __name__ == "__main__":
    # load data
    pushing_multistep_residual_dynamics_model = None
    collected_data = np.load("collected_data.npy", allow_pickle=True) 
    train_loader, val_loader = process_data_multiple_step(collected_data, 
                                                          batch_size=BATCH_SIZE, 
                                                          num_steps=NUM_STEPS)

    # loss function
    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = MultiStepLoss(pose_loss, discount=0.9)

    # create model
    env = PandaPushingEnv()
    env.reset()
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    if MODEL == "ode":
        dynamics_model = NeuralODEDynamicsModel(state_dim, action_dim).to(DEVICE)
    elif MODEL == "residual_nn":
        dynamics_model = ResidualDynamicsModel(state_dim, action_dim).to(DEVICE)

    # training
    train_losses, val_losses = train_model(dynamics_model, 
                                           train_loader, 
                                           val_loader, 
                                           num_epochs=NUM_EPOCHS, 
                                           lr=LR
                                           )

    # plot train loss and test loss
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
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
    plt.show()
    plt.savefig('train_val_loss.png')

    # save model
    if MODEL == "ode":
        save_path = f'checkpoint/pushing_{NUM_STEPS}_steps_ode_dynamics_model.pt'
    elif MODEL == "residual_nn":
        save_path = f'checkpoint/pushing_{NUM_STEPS}_steps_residual_dynamics_model.pt'
    torch.save(dynamics_model.state_dict(), save_path)