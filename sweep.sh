#!/bin/bash
#SBATCH -J ae-latent-sweep
#SBATCH -A plgcompneurosano2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --array=1-9
#SBATCH -o logs/latent-dims/latent_%a_%j.out
#SBATCH -e logs/latent-dims/latent_%a_%j.err

# Latent dimension sweep: zscore normalisation + cosine scheduler (chosen from baseline).
# 9 jobs: 1=2  2=4  3=6  4=8  5=12  6=16  7=32  8=64  9=128
# Normal usage: sbatch sweep.sh
#
# To run a single dim manually (no array):
#   sbatch --export=ALL,LATENT_DIM=32 sweep.sh

module purge
module load GCCcore/13.3.0
module load Python/3.12.3
module load CUDA/12.8.0

source /net/tscratch/people/plgjuliaryb/envs/dl-tract-density-survival/bin/activate
echo "loaded venv"

export WANDB_MODE=offline
export WANDB_DIR=/net/tscratch/people/plgjuliaryb/data/dl-tract-density-survival-outputs
echo "exported wandb env vars"

# Map array task ID -> latent dimension
# 1=2  2=4  3=6  4=8  5=12  6=16  7=32  8=64  9=128
LATENT_DIMS=( 2 4 6 8 12 16 32 64 128 )

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
    LATENT_DIM=${LATENT_DIMS[$IDX]}
fi
# If submitted without --array, LATENT_DIM must be set via --export (see above)

echo "Task ${SLURM_ARRAY_TASK_ID:-manual}: latent=$LATENT_DIM norm=zscore sched=cosine"
python train_sweep.py --latent-dim $LATENT_DIM --normalisation zscore

# NORMS=(  none    none                  zscore  zscore                log1p   log1p                log1p_zscore  log1p_zscore        )
# SCHEDS=( ""      "--no-lr-scheduler"   ""      "--no-lr-scheduler"   ""      "--no-lr-scheduler"  ""            "--no-lr-scheduler" )
# # 1=none+cosine  2=none+flat  3=zscore+cosine  4=zscore+flat
# # 5=log1p+cosine 6=log1p+flat 7=log1p_zscore+cosine 8=log1p_zscore+flat

# IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
# NORM=${NORMS[$IDX]}
# SCHED_ARG=${SCHEDS[$IDX]}
# LATENT_DIM=2
# python train_sweep.py --latent-dim $LATENT_DIM --normalisation $NORM $SCHED_ARG

echo "finished"