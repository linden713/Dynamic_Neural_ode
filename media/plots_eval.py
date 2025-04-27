import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data Provided ---
# Parsed from the user's text summary
## == free space ==
results_data = {
    ("Neural Ode", 1): (1, 12.6),
    ("Neural Ode", 2): (1, 7.6),
    ("Neural Ode", 3): (1, 5.4),
    ("Neural Ode", 4): (1, 8.6),
    ("Neural Ode", 5): (1, 7.8),
    ("Residual", 1): (1, 7.8),
    ("Residual", 2): (1, 8.4),
    ("Residual", 3): (1, 12.6),
    ("Residual", 4): (1, 10.8),
    ("Residual", 5): (1, 9.8),
}

## == obstacle ==
# results_data = {
#     ("Neural Ode", 1): (0.20, 17.50),
#     ("Neural Ode", 2): (0.30, 17.40),
#     ("Neural Ode", 3): (0.40, 20.00),
#     ("Neural Ode", 4): (0.20, 17.80),
#     ("Neural Ode", 5): (0.20, 18.30),
#     ("Residual", 1): (0.00, 20.00),
#     ("Residual", 2): (0.10, 18.80),
#     ("Residual", 3): (0.10, 20.00),
#     ("Residual", 4): (0.30, 18.10),
#     ("Residual", 5): (0.10, 19.90),
# }


# Configuration based on the data and previous script style
num_steps_list = [1, 2, 3, 4, 5] # x-axis values
models = ["Neural Ode", "Residual"] # Models included
max_steps_per_episode = 20 # Infer MAX_STEPS from the data
eval_type = "Free Space" # "Obstacle" or "Free Space"
plot_filename = f"media/evaluation_summary_plot_{eval_type}.png" # Output filename

# --- Font Size Configuration ---
title_fontsize = 16
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 11 # Keep legend slightly smaller to fit

# --- Plotting Logic ---

print("Generating results plot (smaller canvas, larger font)...")

# CHANGED: Reduced figsize from (12, 7) to (9, 6)
fig, ax1 = plt.subplots(figsize=(9, 6))

# Define plot styles consistent with the previous script
colors = plt.cm.viridis(np.linspace(0, 1, len(models) * 2))
linestyles_sr = ['-', '-']
markers_sr = ['o', '*']
linestyles_step = ['-.', '-.']
markers_step = ['x', '+']

# --- Axis 1 Setup (Success Rate) ---
# CHANGED: Added fontsize
ax1.set_xlabel('Number of Training Steps used for Dynamics Model', fontsize=label_fontsize)
ax1.set_ylabel('Success Rate', color='tab:blue', fontsize=label_fontsize)
# CHANGED: Added labelsize for ticks
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=tick_fontsize)
ax1.tick_params(axis='x', labelsize=tick_fontsize) # Apply to x-axis ticks too
ax1.set_ylim(0, 1.1)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# --- Axis 2 Setup (Average Steps) ---
ax2 = ax1.twinx()
# CHANGED: Added fontsize
ax2.set_ylabel('Average Steps', color='tab:red', fontsize=label_fontsize)
# CHANGED: Added labelsize for ticks
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=tick_fontsize)

# --- Data Extraction and Plotting ---
min_step_overall = max_steps_per_episode + 1
max_step_overall = 0
lines = []
labels = []

for i, model_name in enumerate(models):
    model_display_name = model_name
    success_rates = []
    avg_steps_list = []
    valid_steps_train = []
    current_model_min_step = max_steps_per_episode + 1
    current_model_max_step = 0

    for num_steps in num_steps_list:
        result = results_data.get((model_name, num_steps))
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
        print(f"Warning: No valid results found for model '{model_display_name}'. Skipping plot lines.")
        continue

    line1, = ax1.plot(valid_steps_train, success_rates,
                      marker=markers_sr[i],
                      linestyle=linestyles_sr[i],
                      color=colors[2*i],
                      label=f'{model_display_name} - Success Rate')
    lines.append(line1)
    labels.append(f'{model_display_name} - Success Rate')

    line2, = ax2.plot(valid_steps_train, avg_steps_list,
                      marker=markers_step[i],
                      linestyle=linestyles_step[i],
                      color=colors[2*i+1],
                      label=f'{model_display_name} - Avg Steps')
    lines.append(line2)
    labels.append(f'{model_display_name} - Avg Steps')

# --- Final Plot Adjustments ---
if min_step_overall > max_steps_per_episode: min_step_overall = 0
ax2.set_ylim(max(0, min_step_overall - 2) , max_step_overall + 2)

# CHANGED: Added fontsize
plt.title(f'Model Performance vs. Training Steps ({eval_type})', fontsize=title_fontsize) # Shortened title slightly
# CHANGED: Added fontsize
ax1.legend(lines, labels, loc='best', fontsize=legend_fontsize)
ax1.set_xticks(num_steps_list)

# IMPORTANT: Use tight_layout() to prevent labels/titles from overlapping
# This is especially crucial with larger fonts on a smaller canvas.
fig.tight_layout()

# --- Save and Show Plot ---
try:
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'")
    plt.show()
except Exception as e:
    print(f"Error saving/showing plot '{plot_filename}': {e}")
finally:
     plt.close(fig)

print("Plot generation complete.")