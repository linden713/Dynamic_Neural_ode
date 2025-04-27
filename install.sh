#!/bin/bash

# set -e  

# if ! command -v conda &> /dev/null
# then
#     echo "❌ Conda not found. Please install Anaconda or Miniconda first!"
#     exit 1
# fi

# source "$(conda info --base)/etc/profile.d/conda.sh"

# if conda env list | grep -q "^rob_learning"; then
#     echo "✅ Conda environment 'rob_learning' already exists. Skipping creation."
# else
#     echo "🛠️  Creating conda environment 'rob_learning'..."
#     conda create -n rob_learning python=3.10 -y
# fi

# conda activate rob_learning
echo "=========================================="
echo "📦 Installing Python packages..."

pip install torch torchdiffeq tqdm numpy matplotlib gym pybullet ipython numpngw pytz rich

echo "🎉 All dependencies installed successfully!"
echo "=========================================="
