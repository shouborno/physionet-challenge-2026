#!/bin/bash
#SBATCH --job-name=pn26_extract
#SBATCH --partition=quick
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-127%64
#SBATCH --output=slurm/logs/extract_%A_%a.out
#SBATCH --error=slurm/logs/extract_%A_%a.out
#
# Feature extraction array. Sized for parallel EDF reads off Ceph rather than
# for cores: extraction is ~13 s of CPU per record but each EDF is ~200 MB, so
# I/O is the binding constraint.
#
# Idempotent: each task skips records whose pickle already exists, so a rerun
# after adding features costs only the new work.
#
# Usage:
#   sbatch --export=ALL,DATA_FOLDER=...,OUTDIR=... slurm/extract.sh

set -o pipefail

PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
export PATH=/home/simran/.conda/envs/pn26/bin:$PATH
export PYTHONPATH="$PROJECT:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT" || exit 1

: "${DATA_FOLDER:?set DATA_FOLDER}"
: "${OUTDIR:?set OUTDIR}"

N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

echo "host=$(hostname) shard=$SHARD/$N_SHARDS start=$(date -Is)"
echo "data=$DATA_FOLDER out=$OUTDIR"

python -u scripts/extract_features.py \
    --data-folder "$DATA_FOLDER" \
    --outdir "$OUTDIR" \
    --shard "$SHARD" \
    --n-shards "$N_SHARDS" \
    ${EXTRA_ARGS:-}
RC=$?

echo "end=$(date -Is) rc=$RC"
exit $RC
