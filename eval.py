import torch
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm
from IPython.display import display
from model.nn_dynamics_models import ResidualDynamicsModel
from model.neural_ODE_dynamics_models import NeuralODEDynamicsModel
from gif.visualizers import GIFVisualizer
from env.panda_pushing_env import PandaPushingEnv
from env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE
from controller import PushingController, free_pushing_cost_function, collision_detection, obstacle_avoidance_pushing_cost_function

# MODEL = "residual_dynamics_model"
MODEL = "neural_ode_dynamics_model"
MAX_STEPS = 20
NUM_STEPS_TRAING = 4

# init visualizer
fig = plt.figure(figsize=(8,8))
hfig = display(fig, display_id=True)
visualizer = GIFVisualizer()

# load the model
env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=True,  camera_heigh=800, camera_width=800)
pushing_model = None
if MODEL == "neural_ode_dynamics_model":
    pushing_neural_ode_dynamics_model = NeuralODEDynamicsModel(
        env.observation_space.shape[0],
        env.action_space.shape[0]
    )
    state_dict = torch.load(f"checkpoint/ode/pushing_{NUM_STEPS_TRAING}_steps_ode_dynamics_model.pt")
    pushing_neural_ode_dynamics_model.load_state_dict(state_dict)
    pushing_model = pushing_neural_ode_dynamics_model.eval()
    
elif MODEL == "residual_dynamics_model":
    pushing_residual_dynamics_model = ResidualDynamicsModel(
        env.observation_space.shape[0],
        env.action_space.shape[0]
    )
    state_dict = torch.load(f"checkpoint/residual/pushing_{NUM_STEPS_TRAING}_steps_residual_dynamics_model.pt")
    pushing_residual_dynamics_model.load_state_dict(state_dict)
    pushing_model = pushing_residual_dynamics_model.eval()

# load pushing environment
env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5)

# load pushing controller - mppi
controller = PushingController(env, pushing_model, free_pushing_cost_function, num_samples=100, horizon=10)

# test the model
state = env.reset()
step = 0
for i in tqdm(range(MAX_STEPS)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    step += 1
    if done:
        break

# evaluate if goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_FREE
goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) 
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')
print(f'GOAL DISTANCE: {goal_distance:.2f}')
print(f'NUMBER OF STEPS: {step}')
visualizer.get_gif() 
plt.close(fig)