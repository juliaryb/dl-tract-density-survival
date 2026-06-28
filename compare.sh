#!/bin/bash
#SBATCH -J ae-compare
#SBATCH -A plgcompneurosano2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o logs/baselines/compare_%j.out
#SBATCH -e logs/baselines/compare_%j.err

module purge
module load GCCcore/13.3.0
module load Python/3.12.3
module load CUDA/12.8.0

source /net/tscratch/people/plgjuliaryb/envs/dl-tract-density-survival/bin/activate

python compare.py --mode baselines
