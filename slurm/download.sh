#!/bin/bash
#SBATCH --job-name=pn26_dl
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/dl_%x_%j.out
#SBATCH --error=slurm/logs/dl_%x_%j.out
#
# Usage: sbatch --job-name=pn26_dl_small slurm/download.sh <dataset-slug> <dest> <manifest>

set -o pipefail

DATASET="$1"
DEST="$2"
MANIFEST="$3"

PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
PY=/home/simran/.conda/envs/pn26/bin/python

cd "$PROJECT" || exit 1

echo "host=$(hostname) start=$(date -Is)"
echo "dataset=$DATASET dest=$DEST"

# The downloader is idempotent, so retry the whole sweep a few times: transient
# Kaggle failures leave gaps that a second pass fills in cheaply (complete files
# are skipped by size comparison).
RC=1
for attempt in 1 2 3; do
    echo "--- sweep $attempt ---"
    "$PY" scripts/download_kaggle.py \
        --dataset "$DATASET" \
        --dest "$DEST" \
        --manifest "$MANIFEST" \
        --workers 6
    RC=$?
    if [ $RC -eq 0 ]; then
        break
    fi
    echo "sweep $attempt exited $RC; retrying after backoff"
    sleep 120
done

echo "end=$(date -Is) rc=$RC"
du -sh "$DEST"
exit $RC
