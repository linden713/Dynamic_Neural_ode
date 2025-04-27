#!/bin/bash

set -e  # 脚本一旦遇到错误，立刻退出！

echo "====== Project Dependency Installer ======"

# 检查 conda 是否安装
if ! command -v conda &> /dev/null
then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first!"
    exit 1
fi

# 初始化 conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# 检查 rob_learning 环境是否存在
if conda env list | grep -q "^rob_learning"; then
    echo "✅ Conda environment 'rob_learning' already exists. Skipping creation."
else
    echo "🛠️  Creating conda environment 'rob_learning'..."
    conda create -n rob_learning python=3.10 -y
fi

# 激活环境
conda activate rob_learning

echo "📦 Installing Python packages..."

pip install torch torchdiffeq tqdm numpy matplotlib gym pybullet ipython numpngw pytz

echo "🎉 All dependencies installed successfully!"
echo "## Follow the instructions in the README to run the project. ##"
echo "=========================================="
