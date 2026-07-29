#!/bin/bash
#SBATCH --job-name=pn26_ps
#SBATCH --partition=short
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%12
#SBATCH --output=slurm/logs/ps_%A_%a.out
#SBATCH --error=slurm/logs/ps_%A_%a.out
#
# Philosopher's Stone latent extraction. Avoids rtx_pro_6000_b (Blackwell,
# sm_120) since torch 2.5.1+cu121 cannot target it.
set -o pipefail
PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
export PATH=/home/simran/.conda/envs/pn26/bin:$PATH
export PYTHONPATH="$PROJECT:$PYTHONPATH"
cd "$PROJECT" || exit 1
: "${DATA_FOLDER:?}"; : "${OUTDIR:?}"
echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID} start=$(date -Is)"
python -u scripts/extract_ps_latents.py \
    --data-folder "$DATA_FOLDER" --outdir "$OUTDIR" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}"
RC=$?
echo "end=$(date -Is) rc=$RC"
exit $RC
