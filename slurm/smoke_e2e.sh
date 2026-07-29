#!/bin/bash
#SBATCH --job-name=pn26_smoke
#SBATCH --partition=quick
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
#
# Full train -> run -> evaluate cycle on a fixture, using the organizers' own
# unmodified train_model.py / run_model.py / helper_code.py. This is the gate
# every submission must pass before it goes out.
#
# Usage: sbatch slurm/smoke_e2e.sh [fixture_dir]

set -o pipefail

PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
export PATH=/home/simran/.conda/envs/pn26/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT" || exit 1

FIX="${1:-data/fixtures/smoke}"
MODEL="$FIX/model"
OUT="$FIX/outputs"

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) fixture=$FIX"

rm -rf "$MODEL" "$OUT"
mkdir -p "$MODEL" "$OUT"

echo "=== TRAIN ==="
python train_model.py -d "$FIX/training_data" -m "$MODEL" -v
RC=$?
if [ $RC -ne 0 ]; then echo "train failed rc=$RC"; exit $RC; fi

echo "=== RUN ==="
python run_model.py -d "$FIX/holdout_data" -m "$MODEL" -o "$OUT" -v
RC=$?
if [ $RC -ne 0 ]; then echo "run failed rc=$RC"; exit $RC; fi

echo "=== EVALUATE ==="
# Predictions land in the updated demographics table written by run_model.
PRED=$(find "$OUT" -name '*.csv' | head -1)
echo "predictions=$PRED"
python evaluate_model.py \
    -d "$FIX/holdout_labels.csv" \
    -o "$PRED" \
    -p "$FIX/prevalence.csv" \
    -s "$FIX/scores.csv" \
    -t "$FIX/table.csv"
RC=$?

echo "=== SCORES ==="
cat "$FIX/scores.csv" 2>/dev/null

echo "end=$(date -Is) rc=$RC"
exit $RC
