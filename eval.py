import torch
import numpy as np
import matplotlib.pyplot as plt


from tqdm.notebook import tqdm
from IPython.display import display
from model.nn_dynamics_models import ResidualDynamicsModel
from gif.visualizers import GIFVisualizer
from env.panda_pushing_env import PandaPushingEnv
from env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE
from controller import PushingController, free_pushing_cost_function, collision_detection, obstacle_avoidance_pushing_cost_function

fig = plt.figure(figsize=(8,8))
hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()

env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=True,  camera_heigh=800, camera_width=800)

# 1. 先实例化模型
pushing_residual_dynamics_model = ResidualDynamicsModel(
    env.observation_space.shape[0],
    env.action_space.shape[0]
)

# 2. 再加载 state_dict
state_dict = torch.load("checkpoint/pushing_multi_step_residual_dynamics_model.pt")
pushing_residual_dynamics_model.load_state_dict(state_dict)

# 3. 切换到评估模式
pushing_residual_dynamics_model.eval()


env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)
controller = PushingController(env, pushing_residual_dynamics_model, free_pushing_cost_function, num_samples=100, horizon=10)

state = env.reset()

num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

        
# Evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_FREE
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')
visualizer.get_gif() 
plt.close(fig)