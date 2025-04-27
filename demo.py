import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import gc

from tqdm import tqdm
from IPython.display import display
from model.nn_dynamics_models import ResidualDynamicsModel
from model.neural_ODE_dynamics_models import NeuralODEDynamicsModel
from gif.visualizers import GIFVisualizer
from env.panda_pushing_env import PandaPushingEnv
from env.panda_pushing_env import TARGET_POSE_FREE, BOX_SIZE
from controller import PushingController, free_pushing_cost_function, obstacle_avoidance_pushing_cost_function

# 配置
MODELS = ["neural_ode_dynamics_model", "residual_dynamics_model"]
NUM_STEPS_TRAIN_LIST = [3, 4]
COST_FUNCTIONS = {
    "free_pushing": free_pushing_cost_function,
    "obstacle_avoidance": obstacle_avoidance_pushing_cost_function,
}
EVAL_EPISODES = 30
MAX_STEPS = 20
USE_GIF = False  # 是否生成 gif 动画
MODE = "paper"  # "paper" or "demo"
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

def evaluate_model(env, model_name, num_steps_train, cost_name, cost_fn, device, visualizer=None, fig=None):
    if model_name == "neural_ode_dynamics_model":
        model = NeuralODEDynamicsModel(env.observation_space.shape[0], env.action_space.shape[0])
        model_path = f"checkpoint/ode/pushing_{num_steps_train}_steps_ode_dynamics_model.pt"
    elif model_name == "residual_dynamics_model":
        model = ResidualDynamicsModel(env.observation_space.shape[0], env.action_space.shape[0])
        model_path = f"checkpoint/residual/pushing_{num_steps_train}_steps_residual_dynamics_model.pt"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    controller = PushingController(env, model, cost_fn, num_samples=100, horizon=10)

    episode_records = []

    for episode in tqdm(range(EVAL_EPISODES), desc=f"{model_name}-{num_steps_train}steps-{cost_name}"):
        state = env.reset()

        for step in range(MAX_STEPS):
            action = controller.control(state)
            state, reward, done, _ = env.step(action)

            if done:
                break

        end_state = env.get_state()
        goal_distance = np.linalg.norm(end_state[:2] - TARGET_POSE_FREE[:2])
        goal_reached = goal_distance < BOX_SIZE

        record = {
            "Episode": episode,
            "Success": int(goal_reached),
            "Steps_Taken": step + 1 if goal_reached else MAX_STEPS,
        }
        episode_records.append(record)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if visualizer is not None and episode == 0:
            visualizer.get_gif()

    if fig is not None:
        plt.close(fig)

    return episode_records

def save_episode_records(episode_records_all, filename):
    df = pd.DataFrame(episode_records_all)
    df.to_csv(filename, index=False)
    print(f"📄 Detailed episode results saved to {filename}")

def plot_episode_curves(df, filename_prefix):
    for cost_name in df['Cost_Function'].unique():
        subset = df[df['Cost_Function'] == cost_name]

        # 分组
        for (model_name, training_steps), model_df in subset.groupby(['Model', 'Training_Steps']):
            model_label = f"{model_name}-{training_steps}steps-{cost_name}"

            # Episode Success Curve
            success_cumsum = model_df['Success'].cumsum()
            episode_idx = model_df['Episode'] + 1  # Episode从1开始画更好看
            success_rate_curve = success_cumsum / episode_idx * 100  # 百分比%

            plt.figure(figsize=(8, 5))
            plt.plot(episode_idx, success_rate_curve, marker='o')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Success Rate (%)')
            plt.title(f"Success Curve: {model_label}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/{filename_prefix}_{model_label}_success_curve.png")
            plt.close()

            # Steps per Episode Curve
            plt.figure(figsize=(8, 5))
            plt.plot(episode_idx, model_df['Steps_Taken'], marker='x')
            plt.xlabel('Episode')
            plt.ylabel('Steps Taken')
            plt.title(f"Steps Curve: {model_label}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/{filename_prefix}_{model_label}_steps_curve.png")
            plt.close()

def main():
    device = torch.device("cpu")
    print(f"🔧 Using device: {device}")

    episode_records_all = []

    # ===== 创建 visualizer 和 fig =====
    fig = None
    visualizer = None
    if USE_GIF or MODE == "demo":
        fig = plt.figure(figsize=(8, 8))
        hfig = display(fig, display_id=True)
        visualizer = GIFVisualizer()

    # ===== 创建一次环境 =====
    env = PandaPushingEnv(
        visualizer=visualizer,
        render_non_push_motions=False,
        camera_heigh=800,
        camera_width=800,
        render_every_n_steps=5
    )

    if MODE == "paper":
        eval_episodes = EVAL_EPISODES
    elif MODE == "demo":
        eval_episodes = 1
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    for model_name in MODELS:
        for num_steps_train in NUM_STEPS_TRAIN_LIST:
            for cost_name, cost_fn in COST_FUNCTIONS.items():
                records = evaluate_model(env, model_name, num_steps_train, cost_name, cost_fn, device, visualizer, fig)

                if MODE == "paper":
                    for rec in records:
                        rec.update({
                            "Model": model_name,
                            "Training_Steps": num_steps_train,
                            "Cost_Function": cost_name
                        })
                    episode_records_all.extend(records)

    if MODE == "paper":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_csv_filename = f"{SAVE_DIR}/detailed_episode_records_{timestamp}.csv"
        save_episode_records(episode_records_all, episode_csv_filename)
        plot_episode_curves(pd.DataFrame(episode_records_all), f"detailed_episode_records_{timestamp}")
    else:
        print("\n🎬 Demo mode: Only saved GIFs, no curve or csv output.\n")

    # ===== 资源清理 =====
    env.close()
    del env
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n====== Detailed Evaluation Completed ======")


if __name__ == "__main__":
    main()
