import torch
import numpy as np
from env.panda_pushing_env import PandaPushingEnv
from datasets import process_data_single_step, process_data_multiple_step, SingleStepDynamicsDataset, MultiStepDynamicsDataset

# Data collection parameters
N = 100 # Number of trajectories
T = 10 # Trajectory length
collected_data = np.load('collected_data.npy', allow_pickle=True)

env = PandaPushingEnv()
env.reset()

batch_size = 500
train_loader, val_loader = process_data_single_step(collected_data, batch_size=batch_size)

# let's check your dataloader

# you should return a dataloader
print('Is the returned train_loader a DataLoader?')
print('Yes' if isinstance(train_loader, torch.utils.data.DataLoader) else 'No')
print('')

# You should have used random split to split your data - 
# this means the validation and training sets are both subsets of an original dataset
print('Was random_split used to split the data?')
print('Yes' if isinstance(train_loader.dataset, torch.utils.data.Subset) else 'No')
print('')

# The original dataset should be of a SingleStepDynamicsDataset
print('Is the dataset a SingleStepDynamicsDataset?')
print('Yes' if isinstance(train_loader.dataset.dataset, SingleStepDynamicsDataset) else 'No')
print('')

# we should see the state, action and next state of shape (batch_size, 3)
for item in train_loader:
    print(f'state is shape {item["state"].shape}')
    print(f'action is shape {item["action"].shape}')
    print(f'next_state is shape {item["next_state"].shape}')
    break

train_loader, val_loader = process_data_multiple_step(collected_data, batch_size=500)

# let's check your dataloader

# you should return a dataloader
print('Is the returned train_loader a DataLoader?')
print('Yes' if isinstance(train_loader, torch.utils.data.DataLoader) else 'No')
print('')

# You should have used random split to split your data - 
# this means the validation and training sets are both subsets of an original dataset
print('Was random_split used to split the data?')
print('Yes' if isinstance(train_loader.dataset, torch.utils.data.Subset) else 'No')
print('')

# The original dataset should be of a MultiStepDynamicsDataset
print('Is the dataset a SingleStepDynamicsDataset?')
print('Yes' if isinstance(train_loader.dataset.dataset, MultiStepDynamicsDataset) else 'No')
print('')

# we should see the state is shape (batch_size, 3)
# and action, next_state are shape (batch_size, num_steps, 3)
for item in train_loader:
    print(f'state is shape {item["state"].shape}')
    print(f'action is shape {item["action"].shape}')
    print(f'next_state is shape {item["next_state"].shape}')
    break