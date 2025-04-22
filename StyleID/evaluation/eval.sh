#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_styleid
#SBATCH --output=logs/eval_styleid_%j.out

# 加载必要模块
module purge
module load StdEnv/2020
module load gcc/9.3.0
module load python/3.8.10
module load cuda/11.4

# 激活虚拟环境
source ~/styleid_env/bin/activate
# rm ./eval_results.txt

# art-fid
echo "开始评估 ArtFID（使用 GPU）"
# python eval_artfid.py --model StyleID --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ../output
# python eval_artfid.py --model Z-Star --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ../../Zero-shot/workdir/test
python eval_artfid.py --model Puff-Net --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ../../Puff-Net/output/test

# histo loss
echo "ArtFID 评估完成。开始评估 histogram loss"
# python eval_histogan.py --model StyleID --sty ../data/sty_eval --tar ../output
# python eval_histogan.py --model Z-Star --sty ../data/sty_eval --tar ../../Zero-shot/workdir/test
python eval_histogan.py --model Puff-Net --sty ../data/sty_eval --tar ../../Puff-Net/output/test