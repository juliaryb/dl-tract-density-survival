#!/bin/bash
# Penalised Cox sweep: 9 latent dims x 4 L1 ratios x 2 penalty variants = 72 cells.
#
#   mkdir -p logs
#   chmod +x sweep_regularisation.sh
#   nohup ./sweep_regularisation.sh > logs/sweep_master.log 2>&1 &
#   tail -f logs/sweep_master.log
#
# Safe to re-run: completed cells are recorded in logs/.done and skipped, so a
# dropped connection or a killed process just means starting it again.

set -u

SCRIPT="latent-components_CV_autoencoder.py"
LOGS="logs"
DONE="$LOGS/.done"

# Set this if clinical_latent_<dim>.csv does not live in the script's default root:
#   ROOT_ARG="--root /path/to/csv/dir"
ROOT_ARG=""

DIMS=(2 4 6 8 12 16 32 64 128)
L1S=(0.1 0.5 0.9 1.0)

mkdir -p "$LOGS" "$DONE"

total=$(( ${#DIMS[@]} * ${#L1S[@]} * 2 ))
n=0; failed=0; skipped=0
start_all=$SECONDS

echo "=== sweep started $(date) : $total cells ==="

for dim in "${DIMS[@]}"; do
  for l1 in "${L1S[@]}"; do
    for demog in "" "--regularize-demographics"; do
      n=$(( n + 1 ))
      if [ -n "$demog" ]; then suffix="pen"; else suffix="exempt"; fi
      tag="dim${dim}_L1${l1}_${suffix}"

      if [ -f "$DONE/$tag" ]; then
        echo "[$n/$total] SKIP    $tag  (already completed)"
        skipped=$(( skipped + 1 ))
        continue
      fi

      echo "[$n/$total] $(date +%H:%M:%S) START   $tag"
      start=$SECONDS

      # $demog and $ROOT_ARG are intentionally unquoted: empty means "no flag".
      if python "$SCRIPT" --latent-dim "$dim" --l1-ratio "$l1" $demog $ROOT_ARG \
             --no-fold-plots > "$LOGS/${tag}.log" 2>&1; then
        touch "$DONE/$tag"
        echo "[$n/$total] $(date +%H:%M:%S) OK      $tag  ($(( SECONDS - start ))s)"
        # Surface the headline numbers so the master log is readable on its own.
        grep -E "Best alpha \(1-SE|Paired vs baseline|latent model better|boundary of the grid|Covariates retained" \
             "$LOGS/${tag}.log" | sed 's/^/            /'
      else
        failed=$(( failed + 1 ))
        echo "[$n/$total] $(date +%H:%M:%S) FAILED  $tag  -> $LOGS/${tag}.log"
        tail -n 3 "$LOGS/${tag}.log" | sed 's/^/            /'
      fi
    done
  done
done

echo
echo "=== sweep finished $(date) ==="
echo "    cells: $total | skipped: $skipped | failed: $failed | elapsed: $(( (SECONDS - start_all) / 60 )) min"
[ "$failed" -gt 0 ] && echo "    re-run this script to retry only the failed cells"
exit 0
