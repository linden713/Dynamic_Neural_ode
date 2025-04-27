import os
import time
import argparse 
import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich import print as rprint 
from model.nn_dynamics_models import ResidualDynamicsModel
from model.neural_ODE_dynamics_models import NeuralODEDynamicsModel
from gif.visualizers import GIFVisualizer
from env.panda_pushing_env import PandaPushingEnv
from env.panda_pushing_env import TARGET_POSE_FREE, BOX_SIZE 
from controller import PushingController, free_pushing_cost_function
import numpy as np

# --- Configuration ---
MODELS_TO_EVAL = ["neural_ode_dynamics_model", "residual_dynamics_model"] 
NUM_STEPS_TRAIN_LIST = [1, 2, 3, 4, 5] 
EVAL_EPISODES = 5           
MAX_STEPS_PER_EPISODE = 20  
ENABLE_GIF_GENERATION = True         # Flag to enable/disable GIF saving
GIF_SAVE_DIR = "gif"                 # Directory to save GIFs
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
            # Log error but allow continuation if possible
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
            # Basic conversion assuming float images are in [0, 1] range
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255).astype(np.uint8)
            else: # Fallback conversion, might need adjustment based on actual image range
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
            # Delay is inter-frame delay in ms
            numpngw.write_apng(filename, self.frames, delay=100)
            console.print(f"\r[green]  Successfully saved animation [bold red]'{filename}'[/]!      [/]")
        except ImportError:
             console.print(f"\r[bold red]Error:[/] [red]Package 'numpngw' not found. Cannot save APNG. Install: pip install numpngw[/]")
             return None
        except Exception as e:
            console.print(f"\r[bold red]Error saving animation '{filename}': {e}[/]")
            return None
        return filename

def evaluate_model(model_name: str, num_steps_train: int, config: dict) -> tuple[float | None, float | None]:
    """
    Evaluates a specified dynamics model for a given training horizon.

    Args:
        model_name (str): Identifier for the model type (e.g., "neural_ode_dynamics_model").
        num_steps_train (int): The training horizon (k-steps) the model was trained for.
        config (dict): Dictionary containing configuration like EVAL_EPISODES, MAX_STEPS, etc.

    Returns:
        tuple[float | None, float | None]: A tuple containing (success_rate, average_steps).
                                            Returns (None, None) if evaluation fails (e.g., model not found).
    """
    # Extract config values
    eval_episodes = config.get('EVAL_EPISODES', 5)
    max_steps = config.get('MAX_STEPS_PER_EPISODE', 20)
    use_gif = config.get('ENABLE_GIF_GENERATION', True)
    gif_dir = config.get('GIF_SAVE_DIR', 'evaluation_gifs')
    ckpt_dir = config.get('CHECKPOINT_BASE_DIR', 'checkpoint')

    visualizer = None
    gif_saved_for_this_run = False

    # Prepare display name for printing
    model_display_name = model_name.replace("_dynamics_model", "").replace("_", " ").title()
    console.print(f"\n[bold bright_blue]===== Evaluating: [bright_magenta]{model_display_name}[/] ([yellow]{num_steps_train}[/] Train Steps) =====")

    # --- Setup Visualizer ---
    if use_gif:
        ensure_dir(gif_dir)
        visualizer = GIFVisualizer()
        console.print(f"[green]GIF generation enabled.[/]")

    # --- Load Environment ---
    # console.print(f"[dim]Loading environment...[/]", end='')
    try:
        env = PandaPushingEnv(
            visualizer=visualizer,
            include_obstacle=INCLUDE_OBSTACLE,
            render_non_push_motions=False, # Optimize rendering if not needed
            camera_heigh=800, camera_width=800,
            render_every_n_steps=5 if use_gif else 0 # Render frequently only if saving GIF
        )
        console.print(f"\r[green]Environment loaded.   [/]")
    except Exception as e:
        console.print(f"\r[bold red]Error loading environment: {e}[/]")
        return None, None

    # --- Load Model ---
    # console.print(f"[dim]Loading model state...[/]", end='')
    model_file_prefix = f"pushing_{num_steps_train}_steps"
    model_path = "" # Initialize model_path
    try:
        # Determine model type and path
        if model_name == "neural_ode_dynamics_model":
            model = NeuralODEDynamicsModel(env.observation_space.shape[0], env.action_space.shape[0])
            sub_dir = "ode"
            model_suffix = "_ode_dynamics_model.pt"
        elif model_name == "residual_dynamics_model":
            model = ResidualDynamicsModel(env.observation_space.shape[0], env.action_space.shape[0])
            sub_dir = "residual"
            model_suffix = "_residual_dynamics_model.pt"
        else:
            # This case should ideally be caught earlier
            console.print(f"\r[bold red]Error: Unknown model name '{model_name}'[/]")
            return None, None

        model_path = os.path.join(ckpt_dir, sub_dir, model_file_prefix + model_suffix)

        # Ensure checkpoint directory exists before trying to load
        ensure_dir(os.path.dirname(model_path))
        # Load the trained model state
        model.load_state_dict(torch.load(model_path))
        console.print(f"\r[green]Model loaded from [italic]{model_path}[/].[/]")
    except FileNotFoundError:
        console.print(f"\r[bold red]Error:[/] [red] Model file not found at [italic]{model_path}[/]. Ensure checkpoints exist.[/]")
        return None, None
    except Exception as e:
        console.print(f"\r[bold red]Error:[/] [red] Failed to load model state from [italic]{model_path}[/]. Reason: {e}[/]")
        return None, None

    model = model.eval() # Set model to evaluation mode

    # --- Initialize Controller ---
    # console.print(f"[dim]Initializing controller...[/]", end='')
    try:
        # Using Model Predictive Control (MPC) with the loaded dynamics model
        controller = PushingController(
            env, model, free_pushing_cost_function, # Cost function for target reaching
            num_samples=100, # Number of action sequences sampled by CEM
            horizon=10       # Planning horizon for MPC
        )
        console.print(f"\r[green]Controller initialized.[/]    ")
    except Exception as e:
        console.print(f"\r[bold red]Error initializing controller: {e}[/]")
        return None, None

    # --- Run Evaluation Episodes ---
    success_count = 0
    steps_list = [] # List to store steps taken per episode

    console.print(f"[cyan]Running {eval_episodes} evaluation episodes...[/]")
    # Setup Rich progress bar
    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None), # Auto width
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )
    # transient=True makes the progress bar disappear after completion
    with Progress(*progress_columns, console=console, transient=True, refresh_per_second=10) as progress:
        task_desc = f"[cyan]Eval {model_display_name} ({num_steps_train} steps)"
        task = progress.add_task(task_desc, total=eval_episodes)

        for episode in range(eval_episodes):
            state = env.reset()
            if use_gif and visualizer:
                visualizer.reset() # Clear frames for the new episode

            done = False
            step_count = 0
            try:
                # Simulate one episode
                for step in range(max_steps):
                    action = controller.control(state) # Get action from MPC
                    state, reward, done, _ = env.step(action) # Apply action to environment
                    step_count = step + 1
                    if done: # Check if environment signaled episode end
                        break
            except Exception as e:
                 # Log error for this episode but continue evaluation if possible
                 progress.console.print(f"[bold red] Error during episode {episode+1}, step {step+1}: {e} [/]")
                 steps_list.append(max_steps) # Count as failure (max steps)
                 progress.update(task, advance=1)
                 continue # Skip to the next episode

            # --- Check Success Condition ---
            end_state = env.get_state()
            target_state = TARGET_POSE_FREE
            # Calculate distance to target (using only x, y position)
            goal_distance = np.linalg.norm(end_state[:2]-target_state[:2])
            # Success if distance is less than the box size (tolerance)
            goal_reached = goal_distance < BOX_SIZE

            if goal_reached:
                success_count += 1
                steps_list.append(step_count)
                # --- GIF Saving Logic ---
                # Save GIF only on the *first* successful episode for this configuration
                if use_gif and visualizer and not gif_saved_for_this_run:
                    gif_filename = os.path.join(gif_dir, f"{model_name}_{num_steps_train}steps_success.gif")
                    # Print GIF saving message *outside* the progress bar context
                    progress.console.print(f"  [green]Saving successful GIF to [bold red]'{gif_filename}'[/][/]")
                    visualizer.get_gif(gif_filename) # Generate and save the GIF
                    gif_saved_for_this_run = True # Mark that GIF has been saved
            else:
                # If goal not reached, record max steps
                steps_list.append(max_steps)

            progress.update(task, advance=1) # Update progress bar

    # --- Calculate and Print Results for this Configuration ---
    success_rate = success_count / eval_episodes if eval_episodes > 0 else 0
    # Average steps, counting failures as max_steps
    avg_steps = np.mean(steps_list) if steps_list else max_steps

    console.print(f"[bold bright_blue]--- Results for: [bright_magenta]{model_display_name}[/] ([yellow]{num_steps_train}[/] steps) ---[/]")
    # Color success rate based on value
    sr_color = 'green' if success_rate > 0.7 else ('yellow' if success_rate > 0.3 else 'red')
    console.print(f"  Success Rate : [bold {sr_color}]{success_rate * 100:>6.2f}%[/]")
    console.print(f"  Average Steps: [bold cyan]{avg_steps:>6.2f}[/] [dim](Failures count as {max_steps} steps)[/]")
    console.print(f"[bright_blue]-----------------------------------------------------[/]")

    return success_rate, avg_steps


def plot_results(results: dict, num_steps_list: list, models: list, max_steps_per_episode: int):
    """
    Generates and saves a plot comparing model performance (success rate and average steps).

    Args:
        results (dict): Dictionary storing results: (model_name, num_steps_train) -> (success_rate, avg_steps).
        num_steps_list (list): List of training horizons (x-axis).
        models (list): List of model identifiers included in the results.
        max_steps_per_episode (int): Used for scaling the y-axis for average steps.
    """
    console.print("\n[bold cyan]Generating results plot...[/]")

    fig, ax1 = plt.subplots(figsize=(12, 7)) # Create figure and primary axis (for success rate)

    # Define plot styles
    # Using matplotlib's colormap here, but could use Rich colors if desired
    colors = plt.cm.viridis(np.linspace(0, 1, len(models) * 2))
    linestyles_sr = ['-', '--', '-.', ':'] # Success Rate linestyles
    markers_sr = ['o', 's', '^', 'D']     # Success Rate markers
    linestyles_step = [':', '-.', '--', '-'] # Average Steps linestyles (distinct)
    markers_step = ['x', '+', '*', 'v']      # Average Steps markers

    # --- Axis 1 Setup (Success Rate) ---
    ax1.set_xlabel('Number of Training Steps used for Dynamics Model')
    ax1.set_ylabel('Success Rate', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1) # Y-axis from 0% to 110%
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7) # Horizontal grid lines

    # --- Axis 2 Setup (Average Steps) ---
    ax2 = ax1.twinx() # Create secondary axis sharing the same x-axis
    ax2.set_ylabel('Average Steps', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # --- Data Extraction and Plotting ---
    min_step_overall = max_steps_per_episode + 1
    max_step_overall = 0
    lines = [] # For combined legend
    labels = [] # For combined legend

    for i, model_name in enumerate(models):
        model_display_name = model_name.replace("_dynamics_model", "").replace("_", " ").title()
        # Extract results for the current model
        success_rates = []
        avg_steps_list = []
        valid_steps_train = [] # x-values for which data exists
        current_model_min_step = max_steps_per_episode + 1
        current_model_max_step = 0

        for num_steps in num_steps_list:
            result = results.get((model_name, num_steps))
            if result is not None: # Check if data exists for this point
                sr, avs = result
                success_rates.append(sr)
                avg_steps_list.append(avs)
                valid_steps_train.append(num_steps)
                # Track min/max steps for dynamic y-axis scaling
                current_model_min_step = min(current_model_min_step, avs)
                current_model_max_step = max(current_model_max_step, avs)

        # Update overall min/max steps
        min_step_overall = min(min_step_overall, current_model_min_step)
        max_step_overall = max(max_step_overall, current_model_max_step)

        # Only plot if there's valid data for this model
        if not valid_steps_train:
            console.print(f"[yellow]Warning: No valid results found for model '{model_display_name}'. Skipping plot lines.[/]")
            continue

        # Plot Success Rate on ax1
        line1, = ax1.plot(valid_steps_train, success_rates,
                          marker=markers_sr[i % len(markers_sr)],
                          linestyle=linestyles_sr[i % len(linestyles_sr)],
                          color=colors[2*i], # Assign unique color
                          label=f'{model_display_name} - Success Rate')
        lines.append(line1)
        labels.append(f'{model_display_name} - Success Rate')

        # Plot Average Steps on ax2
        line2, = ax2.plot(valid_steps_train, avg_steps_list,
                          marker=markers_step[i % len(markers_step)],
                          linestyle=linestyles_step[i % len(linestyles_step)],
                          color=colors[2*i+1], # Assign unique color
                          label=f'{model_display_name} - Avg Steps')
        lines.append(line2)
        labels.append(f'{model_display_name} - Avg Steps')

    # --- Final Plot Adjustments ---
    # Set y-axis limits for average steps dynamically with padding
    if min_step_overall > max_steps_per_episode: min_step_overall = 0 # Handle case if no successes occurred
    ax2.set_ylim(max(0, min_step_overall - 2) , max_step_overall + 2)

    # Set plot title and combined legend
    plt.title('Model Performance vs. Training Steps in Panda Pushing Task')
    ax1.legend(lines, labels, loc='best') # Combine legends from both axes
    ax1.set_xticks(num_steps_list) # Ensure x-ticks align with evaluated points

    fig.tight_layout() # Adjust layout to prevent labels overlapping
    plot_filename = "evaluation_summary_plot.png"
    try:
        plt.savefig(plot_filename) # Save the plot to a file
        console.print(f"\n[green]Plot saved as [bold red]'{plot_filename}'[/][/]")
        plt.show() # Display the plot interactively
    except Exception as e:
        console.print(f"[bold red]Error saving/showing plot '{plot_filename}': {e}[/]")
    finally:
         plt.close(fig) # Ensure the figure is closed after saving/showing

def main():
    """
    Main execution function to run the evaluation loop and generate results.
    """
    # Store results: (model_name, num_steps_train) -> (success_rate, avg_steps)
    results = {}

    # Define evaluation configuration
    # Could potentially load this from a config file or command-line args
    eval_config = {
        'EVAL_EPISODES': EVAL_EPISODES,
        'MAX_STEPS_PER_EPISODE': MAX_STEPS_PER_EPISODE,
        'ENABLE_GIF_GENERATION': ENABLE_GIF_GENERATION,
        'GIF_SAVE_DIR': GIF_SAVE_DIR,
        'CHECKPOINT_BASE_DIR': CHECKPOINT_BASE_DIR
    }

    # Ensure base checkpoint directories exist before starting evaluations
    ensure_dir(os.path.join(CHECKPOINT_BASE_DIR, "ode"))
    ensure_dir(os.path.join(CHECKPOINT_BASE_DIR, "residual"))

    console.print(f"[bold underline bright_white]Starting Model Evaluation[/]")
    start_time = time.time()

    # --- Evaluation Loop ---
    for model_name in MODELS_TO_EVAL:
        for num_steps_train in NUM_STEPS_TRAIN_LIST:
            # Call the evaluation function for each configuration
            success_rate, avg_steps = evaluate_model(
                model_name,
                num_steps_train,
                eval_config
            )
            # Store results only if evaluation was successful
            if success_rate is not None and avg_steps is not None:
                 results[(model_name, num_steps_train)] = (success_rate, avg_steps)
            else:
                 # Error message is printed inside evaluate_model
                 console.print(f"[yellow]Skipping results storage for [magenta]{model_name}[/] ({num_steps_train} steps) due to evaluation error.[/]")

    # --- Final Summary ---
    console.print(f"\n[bold underline red]Final Evaluation Summary[/]")
    if results:
        # Sort results for consistent display order (by model, then steps)
        sorted_results = sorted(results.items(), key=lambda item: (item[0][0], item[0][1]))
        for (model_name, num_steps_train), (success_rate, avg_steps) in sorted_results:
             model_display_name = model_name.replace("_dynamics_model", "").replace("_", " ").title()
             sr_color = 'green' if success_rate > 0.7 else ('yellow' if success_rate > 0.3 else 'red')
             # Print formatted summary line
             console.print(f" [bright_magenta]{model_display_name:<15}[/] Steps: [yellow]{num_steps_train:<2}[/] -> Success: [bold {sr_color}]{success_rate*100:>6.2f}%[/], Avg Steps: [bold cyan]{avg_steps:>6.2f}[/]")

        # --- Plot Generation ---
        # Pass MAX_STEPS for potential y-axis scaling in plot
        plot_results(results, NUM_STEPS_TRAIN_LIST, MODELS_TO_EVAL, MAX_STEPS_PER_EPISODE)
    else:
        # Message if no results were collected
        console.print(f"\n[yellow]No evaluation results were successfully collected. Skipping plot generation.[/]")

    # --- Execution Time ---
    end_time = time.time()
    total_time = end_time - start_time
    console.print(f"\n[bold green]Evaluation finished in {total_time:.2f} seconds.[/]")


if __name__ == "__main__":
    # Entry point of the script
    main()