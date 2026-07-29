#!/bin/bash
#SBATCH --job-name=pn26_bulk
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
#
# Bulk archive download. Kaggle's per-file endpoint 429s after ~1,600 files
# regardless of pacing, so we pull the whole dataset in a single ranged request
# and extract only what is missing.
#
# Usage: sbatch --job-name=pn26_bulk_small slurm/download_bulk.sh <slug> <dest>

set -o pipefail

DATASET="$1"
DEST="$2"

PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
PY=/home/simran/.conda/envs/pn26/bin/python

cd "$PROJECT" || exit 1

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
echo "dataset=$DATASET dest=$DEST"

# The script resumes from a partial archive, so outer retries are cheap.
RC=1
for attempt in $(seq 1 6); do
    echo "--- attempt $attempt $(date -Is) ---"
    "$PY" -u scripts/download_bulk.py \
        --dataset "$DATASET" \
        --dest "$DEST" \
        --workdir data/archives \
        --keep-archive
    RC=$?
    [ $RC -eq 0 ] && break
    echo "attempt $attempt exited $RC; backing off"
    sleep 300
done

echo "files=$(find "$DEST" -type f | wc -l) size=$(du -sh "$DEST" | cut -f1)"
echo "end=$(date -Is) rc=$RC"
exit $RC
