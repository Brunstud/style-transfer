#!/bin/bash
#SBATCH --job-name=puffnet-test
#SBATCH --account=def-yymao
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=./results_extra/%x-%j.out

# 加载必要模块
module purge
module load StdEnv/2020
module load gcc/9.3.0
module load python/3.8.10
module load cuda/11.4

# 激活虚拟环境
source ~/styleid_env/bin/activate

# === 运行训练脚本 ===
python test_batch.py --iter "15000"
