import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich import print as rprint
# Assume these imports are correct and the files exist in the specified structure
from model.nn_dynamics_models import ResidualDynamicsModel
from model.neural_ODE_dynamics_models import NeuralODEDynamicsModel
# from gif.visualizers import GIFVisualizer # Using the modified version below
from env.panda_pushing_env import PandaPushingEnv
from env.panda_pushing_env import TARGET_POSE_FREE, BOX_SIZE
from controller import PushingController, free_pushing_cost_function
import numpy as np

# --- Configuration ---
# Updated list of models including different ODE solvers
MODELS_TO_EVAL = ['ode_dopri5', 'ode_euler', 'ode_rk4', 'residual_dynamics_model']
NUM_STEPS_TRAIN_LIST = [1, 2, 3, 4, 5]
EVAL_EPISODES = 3
MAX_STEPS_PER_EPISODE = 20
ENABLE_GIF_GENERATION = True         # Flag to enable/disable GIF saving
GIF_SAVE_DIR = "gif"      # Directory to save GIFs (renamed to avoid conflict with training gifs)
CHECKPOINT_BASE_DIR = "checkpoint"   # Base directory for model checkpoints
INCLUDE_OBSTACLE = False

# --- Rich Console Initialization ---
console = Console()

# --- Helper Functions ---
def ensure_dir(directory_path: str):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        console.print(f"[dim]Creating directory: [italic]{directory_path}[/][/]")
        try:
            os.makedirs(directory_path)
        except OSError as e:
            console.print(f"[bold red]Error creating directory {directory_path}: {e}[/]")

# --- Modified GIFVisualizer ---
class GIFVisualizer:
    """
    Collects image frames and generates an animated GIF/APNG using numpngw.
    Uses Rich console for printing status messages.
    """
    def __init__(self):
        self.frames = []

    def set_data(self, img: np.ndarray):
        """Appends an image frame, ensuring uint8 format."""
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        self.frames.append(img)

    def reset(self):
        """Clears collected frames."""
        self.frames = []

    def get_gif(self, filename: str = 'output.gif'):
        """Saves collected frames as an APNG file."""
        if not self.frames:
            console.print("[yellow]Warning: No frames collected to generate GIF.[/]")
            return None

        target_dir = os.path.dirname(filename)
        if target_dir:
            ensure_dir(target_dir) # Ensure the directory exists

        console.print(f"[cyan]Creating animation: [white]'{filename}'[/]...[/]", end='')
        try:
            import numpngw # Lazy import if only used here
            numpngw.write_apng(filename, self.frames, delay=100) # delay is inter-frame delay in ms
            console.print(f"\r[green]  Successfully saved animation [bold red]'{filename}'[/]!      [/]")
        except ImportError:
             console.print(f"\r[bold red]Error:[/] [red]Package 'numpngw' not found. Cannot save APNG. Install: pip install numpngw[/]")
             return None
        except Exception as e:
            console.print(f"\r[bold red]Error saving animation '{filename}': {e}[/]")
            return None
        return filename

def get_model_display_name(model_name: str) -> str:
    """Generates a user-friendly display name for the model."""
    if model_name.startswith("ode_"):
        solver = model_name.split('_')[1]
        return f"ODE {solver.capitalize()}"
    elif model_name == "residual_dynamics_model":
        return "Residual"
    else:
        # Fallback for unexpected names
        return model_name.replace("_dynamics_model", "").replace("_", " ").title()

def evaluate_model(model_name: str, num_steps_train: int, config: dict) -> tuple[float | None, float | None]:
    """
    Evaluates a specified dynamics model for a given training horizon.

    Args:
        model_name (str): Identifier for the model type (e.g., "ode_euler", "residual_dynamics_model").
        num_steps_train (int): The training horizon (k-steps) the model was trained for.
        config (dict): Dictionary containing configuration like EVAL_EPISODES, MAX_STEPS, etc.

    Returns:
        tuple[float | None, float | None]: A tuple containing (success_rate, average_steps).
                                            Returns (None, None) if evaluation fails.
    """
    eval_episodes = config.get('EVAL_EPISODES', 5)
    max_steps = config.get('MAX_STEPS_PER_EPISODE', 20)
    use_gif = config.get('ENABLE_GIF_GENERATION', ENABLE_GIF_GENERATION)
    gif_dir = config.get('GIF_SAVE_DIR', 'evaluation_gifs')
    ckpt_dir = config.get('CHECKPOINT_BASE_DIR', 'checkpoint')

    visualizer = None
    gif_saved_for_this_run = False
    model_display_name = get_model_display_name(model_name) # Use helper for display name

    console.print(f"\n[bold bright_blue]===== Evaluating: [bright_magenta]{model_display_name}[/] ([yellow]{num_steps_train}[/] Train Steps) =====")

    if use_gif:
        ensure_dir(gif_dir)
        visualizer = GIFVisualizer()
        console.print(f"[green]GIF generation enabled.[/]")

    try:
        env = PandaPushingEnv(
            visualizer=visualizer,
            include_obstacle=INCLUDE_OBSTACLE,
            render_non_push_motions=False,
            camera_heigh=800, camera_width=800,
            render_every_n_steps=5 if use_gif else 0
        )
        console.print(f"\r[green]Environment loaded.   [/]")
    except Exception as e:
        console.print(f"\r[bold red]Error loading environment: {e}[/]")
        return None, None

    model_path = ""
    try:
        # Determine model type, class, path, and checkpoint suffix
        # Assuming env dims don't change, get them once
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        model_file_prefix = f"pushing_{num_steps_train}_steps"

        if model_name.startswith("ode_"):
            model = NeuralODEDynamicsModel(state_dim, action_dim)
            # Extract solver name for sub_dir and suffix
            solver_name = model_name # e.g., 'ode_euler'
            sub_dir = solver_name
            model_suffix = f"_{solver_name}_dynamics_model.pt" # e.g., _ode_euler_dynamics_model.pt
        elif model_name == "residual_dynamics_model":
            model = ResidualDynamicsModel(state_dim, action_dim)
            sub_dir = "residual"
            model_suffix = "_residual_dynamics_model.pt"
        else:
            console.print(f"\r[bold red]Error: Unknown model name '{model_name}'[/]")
            return None, None

        model_path = os.path.join(ckpt_dir, sub_dir, model_file_prefix + model_suffix)
        checkpoint_dir = os.path.dirname(model_path)
        ensure_dir(checkpoint_dir) # Ensure the specific checkpoint directory exists

        # Load the trained model state
        model.load_state_dict(torch.load(model_path))
        console.print(f"\r[green]Model loaded from [italic]{model_path}[/].[/]")

    except FileNotFoundError:
        console.print(f"\r[bold red]Error:[/] [red] Model file not found at [italic]{model_path}[/]. Ensure checkpoints exist and naming is correct.[/]")
        # Provide more context on expected naming
        console.print(f"[dim]Expected format: checkpoint/[subdirectory]/pushing_[k]_steps_[model_suffix].pt[/]")
        console.print(f"[dim]Subdirectory for '{model_name}' is expected to be '{sub_dir}'[/]")
        console.print(f"[dim]Model suffix for '{model_name}' is expected to be '{model_suffix}'[/]")
        return None, None
    except Exception as e:
        console.print(f"\r[bold red]Error:[/] [red] Failed to load model state from [italic]{model_path}[/]. Reason: {e}[/]")
        return None, None

    model = model.eval()

    try:
        controller = PushingController(
            env, model, free_pushing_cost_function,
            num_samples=100,
            horizon=10
        )
        console.print(f"\r[green]Controller initialized.[/]    ")
    except Exception as e:
        console.print(f"\r[bold red]Error initializing controller: {e}[/]")
        return None, None

    success_count = 0
    steps_list = []

    console.print(f"[cyan]Running {eval_episodes} evaluation episodes...[/]")
    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"), TimeRemainingColumn(), TextColumn("•"), TimeElapsedColumn(),
    )
    with Progress(*progress_columns, console=console, transient=True, refresh_per_second=10) as progress:
        task_desc = f"[cyan]Eval {model_display_name} ({num_steps_train} steps)"
        task = progress.add_task(task_desc, total=eval_episodes)

        for episode in range(eval_episodes):
            state = env.reset()
            if use_gif and visualizer:
                visualizer.reset()

            done = False
            step_count = 0
            try:
                for step in range(max_steps):
                    action = controller.control(state)
                    state, reward, done, _ = env.step(action)
                    step_count = step + 1
                    if done:
                        break
            except Exception as e:
                 progress.console.print(f"[bold red] Error during episode {episode+1}, step {step+1}: {e} [/]")
                 steps_list.append(max_steps)
                 progress.update(task, advance=1)
                 continue

            end_state = env.get_state()
            target_state = TARGET_POSE_FREE
            goal_distance = np.linalg.norm(end_state[:2]-target_state[:2])
            goal_reached = goal_distance < BOX_SIZE

            if goal_reached:
                success_count += 1
                steps_list.append(step_count)
                if use_gif and visualizer and not gif_saved_for_this_run:
                    # Use model_name in filename for clarity
                    gif_filename = os.path.join(gif_dir, f"{model_name}_{num_steps_train}steps_success.gif")
                    progress.console.print(f"  [green]Saving successful GIF to [bold red]'{gif_filename}'[/][/]")
                    visualizer.get_gif(gif_filename)
                    gif_saved_for_this_run = True
            else:
                steps_list.append(max_steps)

            progress.update(task, advance=1)

    success_rate = success_count / eval_episodes if eval_episodes > 0 else 0
    avg_steps = np.mean(steps_list) if steps_list else max_steps

    console.print(f"[bold bright_blue]--- Results for: [bright_magenta]{model_display_name}[/] ([yellow]{num_steps_train}[/] steps) ---[/]")
    sr_color = 'green' if success_rate > 0.7 else ('yellow' if success_rate > 0.3 else 'red')
    console.print(f"  Success Rate : [bold {sr_color}]{success_rate * 100:>6.2f}%[/]")
    console.print(f"  Average Steps: [bold cyan]{avg_steps:>6.2f}[/] [dim](Failures count as {max_steps} steps)[/]")
    console.print(f"[bright_blue]-----------------------------------------------------[/]")

    return success_rate, avg_steps


def plot_results(results: dict, num_steps_list: list, models_to_plot: list, max_steps_per_episode: int, plot_filename: str = "evaluation_summary_plot.png", title_suffix: str = ""):
    """
    Generates and saves a plot comparing performance for a specific subset of models.

    Args:
        results (dict): Dictionary storing all evaluation results: (model_name, num_steps_train) -> (success_rate, avg_steps).
        num_steps_list (list): List of training horizons (x-axis).
        models_to_plot (list): List of model identifiers to include in this specific plot.
        max_steps_per_episode (int): Used for scaling the y-axis for average steps.
        plot_filename (str): The filename to save the plot.
        title_suffix (str): Optional suffix to add to the plot title for context.
    """
    console.print(f"\n[bold cyan]Generating results plot: '{plot_filename}'...[/]")

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Filter results to only include models specified in models_to_plot
    filtered_results = {k: v for k, v in results.items() if k[0] in models_to_plot}
    if not filtered_results:
        console.print(f"[yellow]Warning: No results found for the specified models: {models_to_plot}. Skipping plot '{plot_filename}'.[/]")
        plt.close(fig)
        return

    # Use a consistent color map but subset based on the number of models in this plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_to_plot) * 2))
    linestyles_sr = ['-', '--', '-.', ':']
    markers_sr = ['o', 's', '^', 'D', 'P', '*'] # Added more markers
    linestyles_step = [':', '-.', '--', '-']
    markers_step = ['x', '+', '*', 'v', 'X', 'd'] # Added more markers

    ax1.set_xlabel('Number of Training Steps used for Dynamics Model')
    ax1.set_ylabel('Success Rate', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Steps', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    min_step_overall = max_steps_per_episode + 1
    max_step_overall = 0
    lines = []
    labels = []

    # Iterate through the specific models for this plot
    for i, model_name in enumerate(models_to_plot):
        model_display_name = get_model_display_name(model_name) # Use helper
        success_rates = []
        avg_steps_list = []
        valid_steps_train = []
        current_model_min_step = max_steps_per_episode + 1
        current_model_max_step = 0

        for num_steps in num_steps_list:
            # Look up result in the original full results dictionary
            result = results.get((model_name, num_steps))
            if result is not None:
                sr, avs = result
                success_rates.append(sr)
                avg_steps_list.append(avs)
                valid_steps_train.append(num_steps)
                current_model_min_step = min(current_model_min_step, avs)
                current_model_max_step = max(current_model_max_step, avs)

        min_step_overall = min(min_step_overall, current_model_min_step)
        max_step_overall = max(max_step_overall, current_model_max_step)

        if not valid_steps_train:
            console.print(f"[yellow]Warning: No valid results found for model '{model_display_name}' in this plot. Skipping plot lines.[/]")
            continue

        # Plot Success Rate
        line1, = ax1.plot(valid_steps_train, success_rates,
                          marker=markers_sr[i % len(markers_sr)],
                          linestyle=linestyles_sr[i % len(linestyles_sr)],
                          color=colors[2*i % len(colors)], # Cycle through colors
                          label=f'{model_display_name} - Success Rate')
        lines.append(line1)
        labels.append(f'{model_display_name} - Success Rate')

        # Plot Average Steps
        line2, = ax2.plot(valid_steps_train, avg_steps_list,
                          marker=markers_step[i % len(markers_step)],
                          linestyle=linestyles_step[i % len(linestyles_step)],
                          color=colors[(2*i+1) % len(colors)], # Cycle through colors
                          label=f'{model_display_name} - Avg Steps')
        lines.append(line2)
        labels.append(f'{model_display_name} - Avg Steps')

    if min_step_overall > max_steps_per_episode: min_step_overall = 0
    ax2.set_ylim(max(0, min_step_overall - 2) , max_step_overall + 2)

    plt.title(f'Model Performance vs. Training Steps{title_suffix}')
    ax1.legend(lines, labels, loc='best')
    ax1.set_xticks(num_steps_list)

    fig.tight_layout()
    try:
        plt.savefig(plot_filename)
        console.print(f"\n[green]Plot saved as [bold red]'{plot_filename}'[/][/]")
        # plt.show() # Optionally show plot interactively
    except Exception as e:
        console.print(f"[bold red]Error saving/showing plot '{plot_filename}': {e}[/]")
    finally:
         plt.close(fig)

def main():
    """
    Main execution function to run the evaluation loop and generate specific comparison plots.
    """
    results = {} # Store results: (model_name, num_steps_train) -> (success_rate, avg_steps)

    eval_config = {
        'EVAL_EPISODES': EVAL_EPISODES,
        'MAX_STEPS_PER_EPISODE': MAX_STEPS_PER_EPISODE,
        'ENABLE_GIF_GENERATION': ENABLE_GIF_GENERATION,
        'GIF_SAVE_DIR': GIF_SAVE_DIR,
        'CHECKPOINT_BASE_DIR': CHECKPOINT_BASE_DIR
    }

    # Ensure base checkpoint directories exist for all models being evaluated
    # Extract unique subdirectories needed
    subdirs_to_check = set()
    for model_name in MODELS_TO_EVAL:
        if model_name.startswith("ode_"):
             subdirs_to_check.add(model_name) # e.g., 'ode_euler'
        elif model_name == "residual_dynamics_model":
            subdirs_to_check.add("residual")
        # Add more rules here if other model types are introduced

    for sub_dir in subdirs_to_check:
         ensure_dir(os.path.join(CHECKPOINT_BASE_DIR, sub_dir))

    console.print(f"[bold underline bright_white]Starting Model Evaluation[/]")
    start_time = time.time()

    # --- Evaluation Loop (runs for all models in MODELS_TO_EVAL) ---
    for model_name in MODELS_TO_EVAL:
        for num_steps_train in NUM_STEPS_TRAIN_LIST:
            success_rate, avg_steps = evaluate_model(
                model_name,
                num_steps_train,
                eval_config
            )
            if success_rate is not None and avg_steps is not None:
                 results[(model_name, num_steps_train)] = (success_rate, avg_steps)
            else:
                 console.print(f"[yellow]Skipping results storage for [magenta]{model_name}[/] ({num_steps_train} steps) due to evaluation error.[/]")

    # --- Final Summary ---
    console.print(f"\n[bold underline red]Final Evaluation Summary[/]")
    if results:
        sorted_results = sorted(results.items(), key=lambda item: (item[0][0], item[0][1]))
        for (model_name, num_steps_train), (success_rate, avg_steps) in sorted_results:
             model_display_name = get_model_display_name(model_name)
             sr_color = 'green' if success_rate > 0.7 else ('yellow' if success_rate > 0.3 else 'red')
             console.print(f" [bright_magenta]{model_display_name:<15}[/] Steps: [yellow]{num_steps_train:<2}[/] -> Success: [bold {sr_color}]{success_rate*100:>6.2f}%[/], Avg Steps: [bold cyan]{avg_steps:>6.2f}[/]")

        # --- Specific Plot Generation ---
        # Plot 1: Compare 'ode_euler' and 'residual_dynamics_model'
        plot1_models = ['ode_euler', 'residual_dynamics_model']
        plot_results(results, NUM_STEPS_TRAIN_LIST, plot1_models, MAX_STEPS_PER_EPISODE,
                     plot_filename="evaluation_euler_vs_residual.png",
                     title_suffix=" (Euler vs Residual)")

        # Plot 2: Compare 'ode_dopri5', 'ode_euler', 'ode_rk4'
        plot2_models = ['ode_dopri5', 'ode_euler', 'ode_rk4']
        # Check if all requested models actually have results before plotting
        models_with_results = set(key[0] for key in results.keys())
        plot2_models_present = [m for m in plot2_models if m in models_with_results]
        if len(plot2_models_present) > 1: # Need at least two models to compare
             plot_results(results, NUM_STEPS_TRAIN_LIST, plot2_models_present, MAX_STEPS_PER_EPISODE,
                         plot_filename="evaluation_ode_solvers_comparison.png",
                         title_suffix=" (ODE Solvers Comparison)")
        elif len(plot2_models_present) == 1:
             console.print(f"[yellow]Only one ODE solver ({plot2_models_present[0]}) had results. Skipping ODE comparison plot.[/]")
        else:
            console.print("[yellow]No results found for ODE solvers ('ode_dopri5', 'ode_euler', 'ode_rk4'). Skipping ODE comparison plot.[/]")

    else:
        console.print(f"\n[yellow]No evaluation results were successfully collected. Skipping plot generation.[/]")

    end_time = time.time()
    total_time = end_time - start_time
    console.print(f"\n[bold green]Evaluation finished in {total_time:.2f} seconds.[/]")


if __name__ == "__main__":
    main()