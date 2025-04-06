from learning_uncertain_dynamics import PushingController, free_pushing_cost_function
from nn_model import DynamicsNNEnsemble
from env.panda_pushing_env import TARGET_POSE_FREE, DISK_RADIUS,PandaDiskPushingEnv
import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display
from visualizers import GIFVisualizer, NotebookVisualizer

fig = plt.figure(figsize=(8,8))
hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()

# 初始化模型结构（确保参数和训练时一致）
pushing_nn_ensemble_model = DynamicsNNEnsemble(state_dim=2, action_dim=3, num_ensembles=1)#num_ensembles=10
GOOGLE_DRIVE_PATH = os.path.dirname(os.path.abspath(__file__))
print("当前文件夹路径是：", GOOGLE_DRIVE_PATH)
# 加载模型权重
load_path = os.path.join(GOOGLE_DRIVE_PATH, 'data/dynamics_nn_ensemble_model.pt')
pushing_nn_ensemble_model.load_state_dict(torch.load(load_path))

# 设置为 eval 模式
pushing_nn_ensemble_model.eval()

cost = free_pushing_cost_function
env = PandaDiskPushingEnv(visualizer=visualizer, render_non_push_motions=False,  
                      camera_heigh=800, camera_width=800, render_every_n_steps=5, include_obstacle=False)

controller = PushingController(env, pushing_nn_ensemble_model, 
                               cost, num_samples=100, horizon=10)
env.reset()

state_0 = env.reset()
state = state_0

num_steps_max = 20

env.reset()
for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break


# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_FREE
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < 1.2*DISK_RADIUS

print(f'GOAL REACHED: {goal_reached}')
        
visualizer.get_gif() 
plt.close(fig)
