#!/bin/bash
# =============================================================================
# Qwen3-1.7B QLoRA 微调项目 - 一键环境安装脚本
# =============================================================================
set -e

echo "=========================================="
echo " Qwen3-1.7B QLoRA 微调项目 - 环境安装"
echo "=========================================="

# 检查 conda 环境
CONDA_ENV_NAME="LLMSFTDemo"

if command -v conda &> /dev/null; then
    echo "[INFO] 检测到 conda"

    # 检查是否在目标环境中运行
    if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
        echo "[WARN] 当前不在 ${CONDA_ENV_NAME} 环境中"
        echo "[INFO] 尝试激活 ${CONDA_ENV_NAME} 环境..."
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
            echo "[ERROR] 未找到 ${CONDA_ENV_NAME} 环境，请先创建："
            echo "  conda create -n ${CONDA_ENV_NAME} python=3.10 -y"
            exit 1
        }
    fi
    echo "[OK] 当前环境: ${CONDA_DEFAULT_ENV}"
else
    echo "[WARN] 未检测到 conda，使用当前 Python 环境"
fi

# 检查 Python 版本
echo ""
echo "[INFO] Python 版本:"
python --version

# 升级 pip
echo ""
echo "[INFO] 升级 pip..."
python -m pip install --upgrade pip

# 安装 PyTorch（CUDA 12.1 版本）
echo ""
echo "[INFO] 安装 PyTorch (CUDA 12.1)..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 Flash Attention（可选，提升训练速度）
echo ""
echo "[INFO] 尝试安装 flash-attn（可选）..."
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "[WARN] flash-attn 安装失败，不影响训练（可选组件）"
}

# 安装项目依赖
echo ""
echo "[INFO] 安装项目依赖..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 下载 NLTK 数据
echo ""
echo "[INFO] 下载 NLTK 数据..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
print('[OK] NLTK 数据下载完成')
"

# 验证安装
echo ""
echo "=========================================="
echo " 验证安装"
echo "=========================================="
python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA 可用:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本:     {torch.version.cuda}')
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    print(f'GPU 显存:      {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

import transformers
print(f'Transformers:  {transformers.__version__}')

import peft
print(f'PEFT:          {peft.__version__}')

import trl
print(f'TRL:           {trl.__version__}')

import bitsandbytes
print(f'BitsAndBytes:  {bitsandbytes.__version__}')

import datasets
print(f'Datasets:      {datasets.__version__}')
"

echo ""
echo "=========================================="
echo " 安装完成！"
echo "=========================================="
