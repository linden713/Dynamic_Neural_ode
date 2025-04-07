import os
import numpy as np
from torch.utils.data import DataLoader
from dataset import SingleStepDynamicsDataset
from nn_model import DynamicsNNEnsemble
from torch.nn import functional as F
from tqdm import tqdm
import torch.optim as optim
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
print("当前文件夹路径是：", current_dir)

train_data = np.load(os.path.join(current_dir, 'data/pushing_training_data.npy'), allow_pickle=True)
validation_data = np.load(os.path.join(current_dir, 'data/pushing_validation_data.npy'), allow_pickle=True)

# Datasets and Dataloaders
train_dataset = SingleStepDynamicsDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
validation_dataset = SingleStepDynamicsDataset(validation_data)
val_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset))
print("Train dataset size: ", len(train_dataset))

# Train the dynamics model
pushing_nn_ensemble_model = DynamicsNNEnsemble(state_dim=2, action_dim=3, num_ensembles=1)#num_ensembles=10


# --- Your code here
optimizer = optim.Adam(pushing_nn_ensemble_model.parameters(), lr=0.0005)
num_epochs = 2000
pushing_nn_ensemble_model.train()
tq = tqdm(range(num_epochs))
tq.set_description("Training dynamics model")

for epoch in tq:
    for batch in train_loader:
        state = batch['state']
        action = batch['action']
        target_next_state = batch['next_state']
        
        optimizer.zero_grad()
        pred_next_state_ensemble = pushing_nn_ensemble_model(state, action)  # shape: (B, N, dx)
        
        # MSE loss averaged across ensemble members
        loss = F.mse_loss(pred_next_state_ensemble, target_next_state.unsqueeze(1).expand_as(pred_next_state_ensemble))
        loss.backward()
        optimizer.step()

        tq.set_postfix(loss=loss.item())
# ---

# save model:
save_path = os.path.join(current_dir, 'checkpoints/dynamics_nn_ensemble_model.pt')
torch.save(pushing_nn_ensemble_model.state_dict(), save_path)