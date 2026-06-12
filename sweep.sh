#!/bin/bash
#SBATCH -J ae-sweep
#SBATCH -A plgcompneurosano2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o logs/latent%a_%j.out
#SBATCH -e logs/latent%a_%j.err


module purge
module load GCCcore/13.3.0
module load Python/3.12.3
module load CUDA/12.8.0

source /net/tscratch/people/plgjuliaryb/envs/dl-tract-density-survival/bin/activate
echo "loaded venv"

# after run is complete sync to online from login node using wandb sync [run-dir]
export WANDB_MODE=offline
export WANDB_DIR=/net/tscratch/people/plgjuliaryb/data/dl-tract-density-survival-outputs
echo "exported env variables for wandb"

python train_sweep.py --latent-dim 4
echo "finished"
