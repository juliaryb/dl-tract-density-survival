#!/bin/bash
#SBATCH -J ae-cox
#SBATCH -A plgcompneurosano2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o logs/cox_%j.out
#SBATCH -e logs/cox_%j.err

module purge
module load GCCcore/13.3.0
module load Python/3.12.3
module load CUDA/12.8.0

source /net/tscratch/people/plgjuliaryb/envs/dl-tract-density-survival/bin/activate

python cox.py
