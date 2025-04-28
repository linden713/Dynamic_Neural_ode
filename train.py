import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model.losses import MultiStepLoss, SE2PoseLoss
from datasets import process_data_multiple_step
from tqdm import tqdm
import torch.optim as optim
from model.nn_dynamics_models import ResidualDynamicsModel
from env.panda_pushing_env import PandaPushingEnv
from model.neural_ODE_dynamics_models import NeuralODEDynamicsModel
import os

# parameters
# MODEL = "residual" 
MODEL = "ode"
LR = 0.0001
NUM_EPOCHS = 3000
NUM_STEPS = 3
BATCH_SIZE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoint" 
PLOT_DIR = "media/plots"
ODE_METHODS_LIST = ['rk4', 'euler', 'dopri5']
NUM_STEPS_LIST = [1, 2, 3, 4, 5] 

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
        next_state_gth = batch_data['next_state'].to(DEVICE) # Ground truth next states over NUM_STEPS
        optimizer.zero_grad()
        # The loss function MultiStepLoss internally calls the model multiple times
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
            next_state_gth = batch_data['next_state'].to(DEVICE) # Ground truth next states over NUM_STEPS
            # The loss function MultiStepLoss internally calls the model multiple times
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
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")

    for epoch_i in pbar:
        train_loss_i = train_step(model, train_dataloader, optimizer, loss_fn)
        val_loss_i = val_step(model, val_dataloader, loss_fn)

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        # Update progress bar description
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')

        # --- Save Best Model ---
        if val_loss_i < best_val_loss:
            print(f"\nEpoch {epoch_i+1}: New best model saved with validation loss: {val_loss_i:.4f}")
            best_val_loss = val_loss_i
            torch.save(model.state_dict(), save_path)
            # Optional: Add an indicator to the progress bar or print a message
            pbar.set_postfix_str(f'Best val loss: {best_val_loss:.4f} (saved)', refresh=True)
            # print(f"Epoch {epoch_i+1}: New best model saved with validation loss: {best_val_loss:.4f}")
        else:
             pbar.set_postfix_str(f'Best val loss: {best_val_loss:.4f}', refresh=True)


    print(f"\nTraining finished. Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_path}")
    return train_losses, val_losses


if __name__ == "__main__":
    for NUM_STEPS in NUM_STEPS_LIST:
        # --- Load Data Once ---
        # Data loading and processing depend only on NUM_STEPS, not the ODE method
        print(f"Loading data for NUM_STEPS = {NUM_STEPS}...")
        collected_data = np.load("data/collected_data.npy", allow_pickle=True)
        train_loader, val_loader = process_data_multiple_step(collected_data,
                                                            batch_size=BATCH_SIZE,
                                                            num_steps=NUM_STEPS)
        print("Data loaded.")

        # --- Environment Info ---
        env = PandaPushingEnv()
        env.reset()
        state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        print(f"State Dim: {state_dim}, Action Dim: {action_dim}")

        # --- Loop Over ODE Integration Methods ---
        for ode_method in ODE_METHODS_LIST:
            print(f"\n===== Training with ODE Method: {ode_method} =====")
            MODEL_TYPE = f"ode_{ode_method}" # Use this for naming outputs

            # --- Loss Function ---
            # Re-initialize loss for safety, although it might be stateless
            pose_loss_func = SE2PoseLoss(block_width=0.1, block_length=0.1)
            # MultiStepLoss applies the model recursively 
            multi_step_loss = MultiStepLoss(pose_loss_func, discount=0.9) 
            # --- Create Model ---
            # Create a new model instance for each ODE method
            dynamics_model = NeuralODEDynamicsModel(state_dim, action_dim, ode_method=ode_method).to(DEVICE)
            print(f"Created Neural ODE model with method: {ode_method}")

            # --- Define Save Paths ---
            # Include the ODE method in the filenames
            method_checkpoint_dir = os.path.join(CHECKPOINT_DIR, MODEL_TYPE)
            method_plot_dir = os.path.join(PLOT_DIR, MODEL_TYPE)

            # Create specific directories for this method if they don't exist
            os.makedirs(method_checkpoint_dir, exist_ok=True)
            os.makedirs(method_plot_dir, exist_ok=True)

            model_save_path = os.path.join(method_checkpoint_dir, f'pushing_{NUM_STEPS}_steps_{MODEL_TYPE}_dynamics_model.pt')
            plot_save_path = os.path.join(method_plot_dir, f'train_val_loss_{MODEL_TYPE}_{NUM_STEPS}_steps.png')
            csv_save_path = os.path.join(method_plot_dir, f'loss_history_{MODEL_TYPE}_{NUM_STEPS}_steps.csv')

            print(f"Model save path: {model_save_path}")
            print(f"Plot save path: {plot_save_path}")
            print(f"CSV save path: {csv_save_path}")

            # --- Training ---
            train_losses, val_losses = train_model(dynamics_model,
                                                train_loader,
                                                val_loader,
                                                loss_fn=multi_step_loss, # Use the multi-step loss
                                                save_path=model_save_path,
                                                num_epochs=NUM_EPOCHS,
                                                lr=LR
                                                )

            # --- Save Final Model State (Optional, train_model already saves the best) ---
            # You might want to save the *final* model state regardless of validation performance
            # final_model_save_path = os.path.join(method_checkpoint_dir, f'pushing_{NUM_STEPS}_steps_{MODEL_TYPE}_dynamics_model_final.pt')
            # torch.save(dynamics_model.state_dict(), final_model_save_path)
            # print(f"Final model state saved to: {final_model_save_path}")


            # --- Save Loss History CSV ---
            loss_data = {
                'epoch': list(range(1, len(train_losses) + 1)), # Epoch numbers (starting from 1)
                'train_loss': train_losses,
                'validation_loss': val_losses
            }
            loss_df = pd.DataFrame(loss_data)

            # Save to CSV
            loss_df.to_csv(csv_save_path, index=False)
            print(f"Loss history saved to: {csv_save_path}")

            # --- Plot Train/Validation Loss ---
            plt.figure(figsize=(10, 5)) # Create a new figure for each method
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.yscale('log') # Use log scale for y-axis
            plt.title(f'Training & Validation Loss (ODE Method: {ode_method}, {NUM_STEPS} steps)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (Log Scale)')
            plt.legend() # Show legend
            plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Grid for log scale
            plt.tight_layout() # Adjust layout
            plt.savefig(plot_save_path) # Save the plot
            print(f"Loss plot saved to: {plot_save_path}")
            plt.close() # Close the plot to free memory

    print("\n===== All training runs completed. =====")

    