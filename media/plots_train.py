import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np # For linspace with colormap

# --- Configuration ---
BASE_DATA_DIR = 'plots' # Base directory containing model subdirectories with CSVs
MODELS = ['ode_euler', 'residual'] # List of model types to process
STEPS = [1, 2, 3, 4, 5] # List of step configurations for each model
# Add smoothing window size configuration
SMOOTHING_WINDOW = 12 # Window size for moving average on training loss

# Update filename to reflect smoothing
PLOT_FILENAME = f"loss_curves_comparison.png" # Output plot filename

# --- Font Size Configuration ---
title_fontsize = 14
suptitle_fontsize = 16
label_fontsize = 12
tick_fontsize = 10
legend_fontsize = 9

# --- Plotting Setup ---
# Create 1 row, 2 columns of subplots, sharing the x-axis
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Define colors (based on steps) and linestyles (based on model)
colors = plt.cm.viridis(np.linspace(0, 1, len(STEPS))) # Colormap for different steps
linestyles = {'ode': '-', 'residual': '--'} # Linestyles for different models

# --- Data Loading and Plotting Loop ---
print(f"Loading data and generating log-scale loss plots ...")
file_found_count = 0 # Counter for successfully processed files

for model in MODELS:
    linestyle = linestyles.get(model, '-') # Get linestyle for the current model
    for i, step in enumerate(STEPS):
        color = colors[i] # Get color for the current step
        # Construct the filename and full path for the CSV
        filename = f"loss_history_{model}_{step}_steps.csv"
        filepath = os.path.join(BASE_DATA_DIR, model, filename)

        # Create a display-friendly model name and legend label
        model_display_name = model.replace("_", " ").title()
        label = f"{model_display_name} - {step} Step{'s' if step > 1 else ''}"

        try:
            # Read the CSV data into a pandas DataFrame
            df = pd.read_csv(filepath)

            # Check for empty DataFrame or non-positive loss values (log scale requires positive values)
            if df.empty or df['train_loss'].min() <= 0 or df['validation_loss'].min() <= 0:
                 print(f"Warning: Skipping {filepath} due to empty data or non-positive loss values required for log scale.")
                 continue # Skip to the next file

            file_found_count += 1 # Increment counter if file is valid

            # --- Apply smoothing (only to training loss) ---
            # Use a centered moving average for smoothing to reduce phase lag
            smoothed_train_loss = df['train_loss'].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=2).mean()

            # --- Plotting on Subplots ---
            # Subplot 1: Smoothed training loss
            axes[0].plot(df['epoch'], smoothed_train_loss, # Use smoothed data
                         label=label,
                         color=color,
                         linestyle=linestyle)

            # Subplot 2: Original validation loss (not smoothed)
            axes[1].plot(df['epoch'], df['validation_loss'],
                         label=label,
                         color=color,
                         linestyle=linestyle)

        except FileNotFoundError:
            print(f"Warning: File not found - {filepath}")
        except KeyError as e:
            # Handle cases where expected columns are missing
            print(f"Warning: Column {e} not found in {filepath}. Skipping file.")
        except Exception as e:
            # Catch any other unexpected errors during file processing
            print(f"Warning: Error processing file {filepath}: {e}")

# --- Final Plot Configuration ---
if file_found_count > 0: # Only configure and save if data was plotted
    # Subplot 0: Training Loss Configuration
    axes[0].set_title(f'Training Loss vs. Epoch (Log Scale)', fontsize=title_fontsize) # Update title
    axes[0].set_ylabel('Training Loss (Log Scale)', fontsize=label_fontsize) # Update Y-axis label
    axes[0].grid(True, which='both', linestyle='--', alpha=0.6) # Add grid lines
    axes[0].tick_params(axis='y', labelsize=tick_fontsize) # Set y-axis tick label size
    axes[0].tick_params(axis='x', labelsize=tick_fontsize) # Set x-axis tick label size
    axes[0].set_yscale('log') # Set y-axis to logarithmic scale
    axes[0].legend(fontsize=legend_fontsize, loc='upper right') # Add legend

    # Subplot 1: Validation Loss Configuration (remains largely the same)
    axes[1].set_title('Validation Loss vs. Epoch (Log Scale)', fontsize=title_fontsize)
    axes[1].set_ylabel('Validation Loss (Log Scale)', fontsize=label_fontsize)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.6) # Add grid lines
    axes[1].tick_params(axis='y', labelsize=tick_fontsize) # Set y-axis tick label size
    axes[1].set_yscale('log') # Set y-axis to logarithmic scale
    axes[1].legend(fontsize=legend_fontsize, loc='upper right') # Add legend

    # Common X-axis Label
    fig.text(0.5, 0.02, 'Epoch', ha='center', va='center', fontsize=label_fontsize)
    # Overall Figure Title
    fig.suptitle('Dynamics Model Training Loss Curves', fontsize=suptitle_fontsize, y=0.98) # Update main title

    # Adjust layout to prevent overlapping elements
    # rect=[left, bottom, right, top] defines the bounding box for tight_layout
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    # --- Save and Display the Plot ---
    try:
        plt.savefig(PLOT_FILENAME) # Save the figure to a file
        print(f"\nPlot saved as '{PLOT_FILENAME}'")
        plt.show() # Display the plot
    except Exception as e:
        print(f"Error saving/showing plot '{PLOT_FILENAME}': {e}")
    finally:
        # Close the figure to free up memory, regardless of success/failure
        plt.close(fig)
else:
    # Message if no valid data files were found or processed
    print("\nNo data files were found or processed (or data contained non-positive values). Skipping plot generation.")

print("\nPlot generation attempt finished.")
