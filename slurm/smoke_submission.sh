#!/bin/bash
#SBATCH --job-name=pn26_smoke
#SBATCH --partition=short,long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
#
# End-to-end rehearsal of the submission using the organizers' own train_model.py
# and run_model.py, unmodified. The previous entry could not have scored because
# team_code.py called a helper deleted upstream, and that was invisible locally
# precisely because our own harness was used instead of theirs.
set -o pipefail
PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
export PATH=/home/simran/.conda/envs/pn26/bin:$PATH
# Only the project. TabFM is now installed into the environment itself.
# Putting its old pip --target directory on the path shadowed the whole
# environment, pandas included, and the organizers' run_model.py died writing a
# prediction into a bool column against the wrong pandas version.
export PYTHONPATH="$PROJECT:$PYTHONPATH"
N=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N
cd "$PROJECT" || exit 1

DATA=${DATA:-/scratch/simran/pn26/raw/training_set_small}
WORK=/scratch/simran/pn26/smoke_${SLURM_JOB_ID:-manual}
rm -rf "$WORK"; mkdir -p "$WORK/model" "$WORK/out"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null|head -1)"
echo "data=$DATA start=$(date -Is)"

# The organizers substitute their own copies of these, so use them verbatim.
echo "=== train_model.py (theirs) ==="
python -u train_model.py -d "$DATA" -m "$WORK/model" -v
RC=$?; [ $RC -ne 0 ] && { echo "TRAIN FAILED rc=$RC"; exit $RC; }
ls -la "$WORK/model"

echo "=== run_model.py (theirs) ==="
python -u run_model.py -d "$DATA" -m "$WORK/model" -o "$WORK/out" -v
RC=$?; [ $RC -ne 0 ] && { echo "RUN FAILED rc=$RC"; exit $RC; }

# evaluate_model.py takes files, not directories, and needs a prevalence file:
# the age-conditioned reward is scored against prevalence estimated from a
# labelled set, for which the training demographics stand in here.
echo "=== evaluate_model.py (theirs) ==="
python -u evaluate_model.py \
    -d "$DATA/demographics.csv" \
    -o "$WORK/out/demographics.csv" \
    -p "$DATA/demographics.csv" \
    -s "$WORK/scores.csv"
RC=$?
cat "$WORK/scores.csv" 2>/dev/null
echo "end=$(date -Is) rc=$RC"
exit $RC
