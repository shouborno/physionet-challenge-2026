#!/bin/bash
#SBATCH --job-name=pn26_run
#SBATCH --partition=quick
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
#
# Generic CPU job wrapper. Everything computational goes through here rather
# than running on the login node.
#
# Usage:
#   sbatch --job-name=<name> [--cpus-per-task=N] [--mem=NG] [--time=HH:MM:SS] \
#       slurm/run.sh <command> [args...]
#
# Example:
#   sbatch --job-name=pn26_tests slurm/run.sh \
#       python -m pytest tests/test_challenge_metrics.py -q

set -o pipefail

PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
export PATH=/home/simran/.conda/envs/pn26/bin:$PATH
export PYTHONPATH="$PROJECT:$PYTHONPATH"

# Keep BLAS from oversubscribing the allocation; joblib does the parallelism.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT" || exit 1

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK start=$(date -Is)"
echo "cmd: $*"
echo "---"

"$@"
RC=$?

echo "---"
echo "end=$(date -Is) rc=$RC"
exit $RC
