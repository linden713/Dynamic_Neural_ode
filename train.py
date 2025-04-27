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
import os

# parameters
MODEL = "residual" 
# MODEL = "ode"
LR = 0.0001
NUM_EPOCHS = 3000
NUM_STEPS = 4
BATCH_SIZE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoint" 
PLOT_DIR = "media/plots"
    
def train_step(model, train_loader, optimizer, loss_fn) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :param loss_fn: The loss function to use.
    :return: train_loss <float> representing the average loss among the different mini-batches.
    """
    train_loss = 0.
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        state = batch_data['state'].to(DEVICE)
        action = batch_data['action'].to(DEVICE)
        next_state_gth = batch_data['next_state'].to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model, state, action, next_state_gth)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def val_step(model, val_loader, loss_fn) -> float:
    """
    Performs an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param val_loader: Pytorch DataLoader
    :param loss_fn: The loss function to use.
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0.
    model.eval()
    with torch.no_grad(): # Disable gradient calculation for validation
        for batch_idx, batch_data in enumerate(val_loader):
            state = batch_data['state'].to(DEVICE)
            action = batch_data['action'].to(DEVICE)
            next_state_gth = batch_data['next_state'].to(DEVICE)
            loss = loss_fn(model, state, action, next_state_gth)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_model(model, train_dataloader, val_dataloader, loss_fn, save_path, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Uses Adam optimizer.
    Saves the best model based on validation loss.
    Creates the save directory if it doesn't exist.

    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param loss_fn: The loss function to use.
    :param save_path: Path to save the best model checkpoint.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return: (train_losses, val_losses) lists containing the loss per epoch.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pbar = tqdm(range(num_epochs), desc="Epochs")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') # Initialize best validation loss to infinity

    # --- Create checkpoint directory if it doesn't exist ---
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True) 

    for epoch_i in pbar:
        train_loss_i = train_step(model, train_dataloader, optimizer, loss_fn)
        val_loss_i = val_step(model, val_dataloader, loss_fn)

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        # Update progress bar description
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')

        # --- Save Best Model ---
        if val_loss_i < best_val_loss:
            print(f"Epoch {epoch_i+1}: New best model saved with validation loss: {val_loss_i:.4f}")
            best_val_loss = val_loss_i
            torch.save(model.state_dict(), save_path)
            # Optional: Add an indicator to the progress bar or print a message
            pbar.set_postfix_str(f'Best val loss: {best_val_loss:.4f} (saved)', refresh=True)
            # print(f"Epoch {epoch_i+1}: New best model saved with validation loss: {best_val_loss:.4f}")

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_path}")
    return train_losses, val_losses


if __name__ == "__main__":
    # load data
    pushing_multistep_residual_dynamics_model = None
    collected_data = np.load("data/collected_data.npy", allow_pickle=True) 
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
    elif MODEL == "residual":
        dynamics_model = ResidualDynamicsModel(state_dim, action_dim).to(DEVICE)
    print(f"Created model: {MODEL}")
    
    # Define save path for the best model
    model_save_path = os.path.join(CHECKPOINT_DIR, f'{MODEL}/pushing_{NUM_STEPS}_steps_{MODEL}_dynamics_model.pt')
    plot_save_path = os.path.join(PLOT_DIR, f'{MODEL}/train_val_loss_{MODEL}_{NUM_STEPS}_steps.png')

    # training
    train_losses, val_losses = train_model(dynamics_model,
                                           train_loader,
                                           val_loader,
                                           loss_fn=pose_loss, # Pass the loss function
                                           save_path=model_save_path, # Pass the save path
                                           num_epochs=NUM_EPOCHS,
                                           lr=LR
                                           )

    # save model
    torch.save(dynamics_model.state_dict(), model_save_path)
    
    # plot train loss and test loss
    plt.figure(figsize=(10, 5)) # Create a single figure
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log') # Use log scale for y-axis
    plt.title(f'Training & Validation Loss ({MODEL}, {NUM_STEPS} steps)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.legend() # Show legend
    plt.grid(True)
    plt.savefig(plot_save_path) # Save the plot to the plots directory
    print(f"Loss plot saved to: {plot_save_path}")
    plt.show() # Display the plot

    