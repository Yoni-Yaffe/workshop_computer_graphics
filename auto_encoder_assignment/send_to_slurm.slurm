#! /bin/sh

#SBATCH --job-name=yoni_test_gpu
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --partition=studentkillable
#SBATCH --time=300
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python train.py
