#!/bin/bash
#SBATCH --job-name=pn26_tp
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-31%24
#SBATCH --output=slurm/logs/tp_%A_%a.out
#SBATCH --error=slurm/logs/tp_%A_%a.out
set -o pipefail
PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
export PATH=/home/simran/.conda/envs/pn26/bin:$PATH
export PYTHONPATH="$PROJECT:$PYTHONPATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT" || exit 1
: "${DATA_FOLDER:?}"; : "${OUTDIR:?}"
echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID} start=$(date -Is)"
python -u scripts/extract_temporal.py --data-folder "$DATA_FOLDER" --outdir "$OUTDIR" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}"
RC=$?
echo "end=$(date -Is) rc=$RC"
exit $RC
