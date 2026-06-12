#!/bin/bash
#SBATCH -J ae-baseline
#SBATCH -A plgcompneurosano2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --array=1-8
#SBATCH -o logs/baseline_%a_%j.out
#SBATCH -e logs/baseline_%a_%j.err

# Baseline comparison: 4 normalisations x 2 scheduler options = 8 parallel jobs.
# Normal usage: sbatch sweep.sh
#
# If --array jobs are unavailable, comment out the #SBATCH --array line above
# and submit each config individually:
#   sbatch --export=ALL,NORM=zscore,SCHED_ARG=""                   sweep.sh
#   sbatch --export=ALL,NORM=zscore,SCHED_ARG="--no-lr-scheduler"  sweep.sh
#   ... (repeat for each norm/sched combination)

module purge
module load GCCcore/13.3.0
module load Python/3.12.3
module load CUDA/12.8.0

source /net/tscratch/people/plgjuliaryb/envs/dl-tract-density-survival/bin/activate
echo "loaded venv"

export WANDB_MODE=offline
export WANDB_DIR=/net/tscratch/people/plgjuliaryb/data/dl-tract-density-survival-outputs
echo "exported wandb env vars"

# Map array task ID -> (normalisation, scheduler flag)
# 1=none+cosine  2=none+flat  3=zscore+cosine  4=zscore+flat
# 5=log1p+cosine 6=log1p+flat 7=log1p_zscore+cosine 8=log1p_zscore+flat
NORMS=(  none    none                  zscore  zscore                log1p   log1p                log1p_zscore  log1p_zscore        )
SCHEDS=( ""      "--no-lr-scheduler"   ""      "--no-lr-scheduler"   ""      "--no-lr-scheduler"  ""            "--no-lr-scheduler" )

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
    NORM=${NORMS[$IDX]}
    SCHED_ARG=${SCHEDS[$IDX]}
fi
# If submitted without --array, NORM and SCHED_ARG must be set via --export (see above)

# Defaults to 2 for the baseline comparison; override with --export=ALL,LATENT_DIM=N
# when reusing this script for the latent sweep after baseline is done
LATENT_DIM=${LATENT_DIM:-2}

echo "Task ${SLURM_ARRAY_TASK_ID:-manual}: norm=$NORM sched='$SCHED_ARG' latent=$LATENT_DIM"
python train_sweep.py --latent-dim $LATENT_DIM --normalisation $NORM $SCHED_ARG
echo "finished"