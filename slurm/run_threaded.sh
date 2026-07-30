#!/bin/bash
#SBATCH --job-name=pn26_thr
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
#
# For single-model workloads that want all cores inside one process, unlike
# slurm/run.sh which pins BLAS to one thread so joblib can fan out safely.
set -o pipefail
PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
export PATH=/home/simran/.conda/envs/pn26/bin:$PATH
export PYTHONPATH="$PROJECT:$PYTHONPATH"
N=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N
cd "$PROJECT" || exit 1
echo "host=$(hostname) threads=$N start=$(date -Is)"
"$@"
RC=$?
echo "end=$(date -Is) rc=$RC"
exit $RC
