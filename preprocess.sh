#!/bin/bash
#SBATCH -J ae-preprocess
#SBATCH -A plgcompneurosano2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 00:02:00
#SBATCH -o logs/preprocess_%j.out
#SBATCH -e logs/preprocess_%j.err

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source /net/tscratch/people/plgjuliaryb/envs/dl-tract-density-survival/bin/activate
echo "loaded venv"

python preprocess_and_cache.py
echo "finished"
