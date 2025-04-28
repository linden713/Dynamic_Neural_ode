import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data Provided (From previous step) ---
## free space
# results_data = {
#     # ODE Dopri5
#     ("ODE Dopri5", 1): (1.00, 8.20),
#     ("ODE Dopri5", 2): (1.00, 7.80),
#     ("ODE Dopri5", 3): (1.00, 8.80),
#     ("ODE Dopri5", 4): (1.00, 8.60),
#     ("ODE Dopri5", 5): (1.00, 9.00),
#     # ODE Euler
#     ("ODE Euler", 1): (1.00, 8.40),
#     ("ODE Euler", 2): (1.00, 7.60),
#     ("ODE Euler", 3): (1.00, 11.00),
#     ("ODE Euler", 4): (1.00, 6.80),
#     ("ODE Euler", 5): (1.00, 11.80),
#     # ODE Rk4
#     ("ODE Rk4", 1): (1.00, 8.00),
#     ("ODE Rk4", 2): (1.00, 6.20),
#     ("ODE Rk4", 3): (1.00, 10.80),
#     ("ODE Rk4", 4): (1.00, 8.40),
#     ("ODE Rk4", 5): (1.00, 10.80),
#     # Residual
#     ("Residual", 1): (1.00, 14.00),
#     ("Residual", 2): (1.00, 13.60),
#     ("Residual", 3): (1.00, 9.60),
#     ("Residual", 4): (1.00, 12.60),
#     ("Residual", 5): (0.80, 15.00),
# }

## obstacle
results_data = {
    # ODE Dopri5
    ("ODE Dopri5", 1): (0.20, 17.50),
    ("ODE Dopri5", 2): (0.30, 17.40), 
    ("ODE Dopri5", 3): (0.40, 20.00), 
    ("ODE Dopri5", 4): (0.20, 17.80), 
    ("ODE Dopri5", 5): (0.20, 18.30), 
    # ODE Euler
    ("ODE Euler", 1): (0.20, 18.30), 
    ("ODE Euler", 2): (0.20, 18.00),  
    ("ODE Euler", 3): (0.30, 20.00),  
    ("ODE Euler", 4): (0.10, 16.40),  
    ("ODE Euler", 5): (0.00, 18.60),  
    # ODE Rk4
    ("ODE Rk4", 1): (0.10, 18.20),   
    ("ODE Rk4", 2): (0.10, 17.20),  
    ("ODE Rk4", 3): (0.40, 18.60),   
    ("ODE Rk4", 4): (0.10, 19.60),   
    ("ODE Rk4", 5): (0.10, 19.40),  
    # Residual
    ("Residual", 1): (0.00, 20.00), 
    ("Residual", 2): (0.10, 18.80),  
    ("Residual", 3): (0.10, 20.00), 
    ("Residual", 4): (0.30, 18.10),  
    ("Residual", 5): (0.10, 19.90),  
}

# Configuration based on the data and previous script style
num_steps_list = [1, 2, 3, 4, 5] # x-axis values
models = ["ODE Dopri5", "ODE Euler", "ODE Rk4", "Residual"] # Models included
max_steps_per_episode = 20 # Set a reasonable default upper bound if needed
eval_type = "Obstacle" # "Obstacle" or "Free Space"

# Ensure the media directory exists
if not os.path.exists("media"):
    os.makedirs("media")
# UPDATED Filename to reflect content
plot_filename = f"media/evaluation_avg_steps_plot_{eval_type}.png"

# --- Font Size Configuration ---
title_fontsize = 16
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 11

# --- Plotting Logic ---

print("Generating average steps plot...")

# Create figure and a single axes object
fig, ax1 = plt.subplots(figsize=(9, 6))

# Define plot styles - Need styles for 4 models
colors = plt.cm.viridis(np.linspace(0, 1, len(models))) # One color per model
linestyles = ['-', '--', ':', '-.'] # 4 distinct styles
markers = ['o', '*', 's', '^']      # 4 distinct markers

# --- Axis Setup (Average Steps Only) ---
ax1.set_xlabel('Number of Training Steps used for Dynamics Model', fontsize=label_fontsize)
ax1.set_ylabel('Average Steps', fontsize=label_fontsize) # Set Y label to Average Steps
ax1.tick_params(axis='y', labelsize=tick_fontsize) # Apply labelsize to Y ticks
ax1.tick_params(axis='x', labelsize=tick_fontsize) # Apply labelsize to X ticks
ax1.grid(True, axis='y', linestyle='--', alpha=0.7) # Keep grid on the Y axis

# --- Data Extraction and Plotting ---
min_step_overall = max_steps_per_episode + 1
max_step_overall = 0
lines = []
labels = []

# Loop through the models list
for i, model_name in enumerate(models):
    model_display_name = model_name
    avg_steps_list = []
    valid_steps_train = []
    current_model_min_step = max_steps_per_episode + 1
    current_model_max_step = 0

    for num_steps in num_steps_list:
        result = results_data.get((model_name, num_steps))
        if result is not None:
            # Only extract average steps (index 1)
            _sr, avs = result # Use _sr to indicate success rate is ignored
            if isinstance(avs, (int, float)): # Check if avg steps value is valid
                avg_steps_list.append(avs)
                valid_steps_train.append(num_steps)
                current_model_min_step = min(current_model_min_step, avs)
                current_model_max_step = max(current_model_max_step, avs)
            else:
                # Handle cases where avg steps might be missing or invalid if necessary
                print(f"Warning: Invalid or missing avg_steps for {model_name}, Steps: {num_steps}")


    min_step_overall = min(min_step_overall, current_model_min_step)
    max_step_overall = max(max_step_overall, current_model_max_step)

    if not valid_steps_train:
        print(f"Warning: No valid results found for model '{model_display_name}'. Skipping plot lines.")
        continue

    # Plot ONLY Average Steps on ax1
    line, = ax1.plot(valid_steps_train, avg_steps_list,
                     marker=markers[i],
                     linestyle=linestyles[i],
                     color=colors[i], # Use color based on index
                     label=model_display_name) # Simplified label for legend
    lines.append(line)
    labels.append(model_display_name) # Add model name to labels list

# --- Final Plot Adjustments ---
# Recalculate y-limits based on actual data range for average steps
if min_step_overall > max_steps_per_episode: min_step_overall = 0 # If no steps found, start at 0
# Set y-limits for average steps on ax1 with a small buffer
ax1.set_ylim(max(0, min_step_overall - 1), max_step_overall + 1)

plt.title(f'Average Steps vs. Training Steps ({eval_type})', fontsize=title_fontsize) # Updated title
ax1.legend(lines, labels, loc='best', fontsize=legend_fontsize) # Legend for the plotted lines
ax1.set_xticks(num_steps_list) # Ensure ticks are at the specified step counts

fig.tight_layout() # Adjust layout to prevent labels overlapping

# --- Save and Show Plot ---
try:
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'")
    plt.show()
except Exception as e:
    print(f"Error saving/showing plot '{plot_filename}': {e}")
finally:
     plt.close(fig) # Close the figure window

print("Average steps plot generation complete.")