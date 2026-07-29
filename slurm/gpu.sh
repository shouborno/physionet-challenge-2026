#!/bin/bash
#SBATCH --job-name=pn26_gpu
#SBATCH --partition=short
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
#
# GPU wrapper. Avoids rtx_pro_6000_b: those are Blackwell (sm_120) and the
# installed torch 2.5.1+cu121 cannot target them.
set -o pipefail
PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
export PATH=/home/simran/.conda/envs/pn26/bin:$PATH
export PYTHONPATH="$PROJECT:$PYTHONPATH"
cd "$PROJECT" || exit 1
echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cmd: $*"
echo "---"
"$@"
RC=$?
echo "---"
echo "end=$(date -Is) rc=$RC"
exit $RC
