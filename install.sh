#!/bin/bash
set -e

# Upgrade pip to the latest version
pip install --upgrade pip

# Install required packages
pip install torch pybullet numpy tqdm numpngw matplotlib torchdiffeq gym pygments

echo "All packages installed successfully."
