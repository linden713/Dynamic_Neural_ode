#!/bin/bash

echo "Installing dependencies for the project..."

conda create -n rob_learning python=3.10 -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate rob_learning

pip install torch torchdiffeq tqdm numpy matplotlib gym pybullet ipython numpngw



echo "Installing dependencies for the project completed."
echo "## Follow the instructions in the README to run the project. ##"