#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=zstar-demo
#SBATCH --output=./results_extra/demo-%x-%j.out

module purge
module load StdEnv/2020  gcc/9.3.0
module load python/3.8.10
module load cuda/11.4
module load opencv/4.6.0
# module load hwloc
# export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/nix/store/z87lf4q1l809fpnmsj9850nb5qxvw2lv-glog-0.3.4/lib:/cvmfs/soft.computecanada.ca/nix/store/q8w6jn55815n83mwyyk31n9wh23a9sds-libtiff-4.0.7/lib:/cvmfs/soft.computecanada.ca/nix/store/k5kqgnlj4jx2khb8hy3wfjf6gmd5f35m-hwloc-1.11.2/lib:/cvmfs/soft.computecanada.ca/nix/store/vdq2wbbm07mzhjj150a8rbxbjc12w71j-user-environment/lib:$LD_LIBRARY_PATH

ENV_DIR=~/style_env
if [ ! -d "$ENV_DIR" ]; then
    python -m venv $ENV_DIR
fi
source $ENV_DIR/bin/activate

python demo.py
