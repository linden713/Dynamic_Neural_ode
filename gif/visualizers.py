import matplotlib.pyplot as plt
import numpngw
import numpy
import os 
import numpy as np
class GIFVisualizer(object):
    def __init__(self):
        self.frames = []

    def set_data(self, img):
        # Ensure image is in uint8 format as expected by numpngw
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255).astype(np.uint8)
            else:
                # Add other potential conversions or raise an error if format is unknown
                img = img.astype(np.uint8) 
        self.frames.append(img)

    def reset(self):
        self.frames = []

    # --- MODIFIED HERE ---
    def get_gif(self, filename='gif/pushing_visualization.gif'): # Add filename parameter
        """Saves the collected frames as an animated PNG (APNG)."""
        if not self.frames:
            print("Warning: No frames collected to generate GIF.")
            return None
            
        # Ensure the target directory exists
        try:
            target_dir = os.path.dirname(filename)
            if target_dir: # Only create if filename includes a directory path
                 os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
             print(f"Warning: Could not create directory {target_dir}. Saving may fail. Error: {e}")


        # generate the gif/apng
        print(f"Creating animated gif/apng: '{filename}', please wait...")
        try:
            # Assuming frames are HxWxC (RGB) or HxW (grayscale), numpngw handles this
            numpngw.write_apng(filename, self.frames, delay=10) # Use the passed filename
            print(f"Successfully saved {filename}")
        except Exception as e:
            print(f"Error saving APNG file '{filename}': {e}")
            return None # Indicate failure
            
        return filename
    # --- END MODIFICATION ---

# --- Keep NotebookVisualizer as it was (seems unrelated to the error) ---
class NotebookVisualizer(object):
    def __init__(self, fig, hfig):
        self.fig = fig
        self.hfig = hfig

    def set_data(self, img):
        plt.clf()
        plt.imshow(img)
        plt.axis('off')
        self.fig.canvas.draw()
        self.hfig.update(self.fig)

    def reset(self):
        pass