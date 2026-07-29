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

SWEEPLOG="slurm/logs/dl_sweeps_${SLURM_JOB_ID}.log"
echo "host=$(hostname) start=$(date -Is) sweeplog=$SWEEPLOG"
echo "dataset=$DATASET dest=$DEST"

# The downloader is idempotent, so retry the whole sweep a few times: transient
# Kaggle failures leave gaps that a second pass fills in cheaply (complete files
# are skipped by size comparison).
RC=1
for attempt in $(seq 1 12); do
    echo "--- sweep $attempt $(date -Is) ---"
    # python -u so progress reaches the log immediately; no pipe filter,
    # since a buffered grep makes a live sweep look hung.
    "$PY" -u scripts/download_kaggle.py \
        --dataset "$DATASET" \
        --dest "$DEST" \
        --manifest "$MANIFEST" \
        --workers "${DL_WORKERS:-3}" \
        --min-interval "${DL_INTERVAL:-0.2}" \
        >> "$SWEEPLOG" 2>&1
    RC=$?
    tail -3 "$SWEEPLOG"
    if [ $RC -eq 0 ]; then
        break
    fi
    # Completed files are skipped by size on the next pass, so a sweep that
    # dies to throttling still makes forward progress. Back off and continue.
    echo "sweep $attempt exited $RC ($(du -sh "$DEST" | cut -f1) so far); backing off"
    sleep 300
done

echo "end=$(date -Is) rc=$RC"
du -sh "$DEST"
exit $RC
