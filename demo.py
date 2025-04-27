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

# 配置
MODELS = ["neural_ode_dynamics_model", "residual_dynamics_model"]
NUM_STEPS_TRAIN_LIST = [3, 4]
EVAL_EPISODES = 5
MAX_STEPS = 20
USE_GIF = False  # 是否生成 gif

def evaluate_model(model_name, num_steps_train):
    if USE_GIF:
        fig = plt.figure(figsize=(8,8))
        hfig = display(fig, display_id=True)
        visualizer = GIFVisualizer()
    else:
        fig = None
        visualizer = None

    # 加载环境
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False, camera_heigh=800, camera_width=800, render_every_n_steps=5)

    # 加载模型
    if model_name == "neural_ode_dynamics_model":
        model = NeuralODEDynamicsModel(env.observation_space.shape[0], env.action_space.shape[0])
        model_path = f"checkpoint/ode/pushing_{num_steps_train}_steps_ode_dynamics_model.pt"
    elif model_name == "residual_dynamics_model":
        model = ResidualDynamicsModel(env.observation_space.shape[0], env.action_space.shape[0])
        model_path = f"checkpoint/residual/pushing_{num_steps_train}_steps_residual_dynamics_model.pt"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    controller = PushingController(env, model, free_pushing_cost_function, num_samples=100, horizon=10)

    success_count = 0
    steps_list = []

    for episode in tqdm(range(EVAL_EPISODES), desc=f"{model_name}-{num_steps_train}steps"):
        state = env.reset()
        for step in range(MAX_STEPS):
            action = controller.control(state)
            state, reward, done, _ = env.step(action)
            if done:
                break
        
        end_state = env.get_state()
        target_state = TARGET_POSE_FREE
        goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) 
        goal_reached = goal_distance < BOX_SIZE

        if goal_reached:
            success_count += 1
            steps_list.append(step+1)  # 步数是从0开始计数的，所以要+1
        else:
            steps_list.append(MAX_STEPS)

        if USE_GIF and episode == 0:  # 只记录第一条GIF，防止生成太多
            visualizer.get_gif()
    
    if fig:
        plt.close(fig)

    success_rate = success_count / EVAL_EPISODES
    avg_steps = np.mean(steps_list)

    return success_rate, avg_steps

def main():
    results = {}

    for model_name in MODELS:
        for num_steps_train in NUM_STEPS_TRAIN_LIST:
            success_rate, avg_steps = evaluate_model(model_name, num_steps_train)
            results[(model_name, num_steps_train)] = (success_rate, avg_steps)

    print("\n====== Evaluation Results ======")
    for (model_name, num_steps_train), (success_rate, avg_steps) in results.items():
        print(f"Model: {model_name}, Training Steps: {num_steps_train}")
        print(f"  Success Rate: {success_rate*100:.2f}%")
        print(f"  Avg Steps to Goal: {avg_steps:.2f}")
        print("--------------------------------")

if __name__ == "__main__":
    main()
