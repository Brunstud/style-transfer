#!/bin/bash
#SBATCH --job-name=styleid-demo
#SBATCH --account=def-yymao
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=./results_extra/%x-%j.out

# 加载必要模块
module purge
module load StdEnv/2020
module load gcc/9.3.0
module load python/3.8.10
module load cuda/11.4

# 激活虚拟环境
source ~/styleid_env/bin/activate

# 执行程序
python run_styleid.py --cnt ../Zero-shot/content_images --sty ../Zero-shot/style_images --output_path ./demo --gamma 0.75 --T 1.5  # default
