#!/bin/bash
#SBATCH --job-name=pn26_reqtest
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
#
# Proxy for the Docker build, which cannot be run here: this cluster has no
# docker, podman or apptainer. A fresh virtualenv on the same Python version as
# the image, installing the exact pinned requirements, catches the two failures
# most likely to break the organizers' build: an unsatisfiable dependency set,
# and the TabFM weight download that the Dockerfile performs at build time.
set -o pipefail
PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
cd "$PROJECT" || exit 1
VENV=/scratch/simran/pn26/reqtest
rm -rf "$VENV"

PY=/home/simran/.conda/envs/pn26/bin/python
echo "base python: $($PY --version)   image targets 3.11"
$PY -m venv "$VENV" || exit 1
source "$VENV/bin/activate"

echo "=== pip install -r requirements.txt ==="
pip install --quiet --upgrade pip
pip install -r requirements.txt 2>&1 | tail -25
RC=${PIPESTATUS[0]}
[ $RC -ne 0 ] && { echo "INSTALL FAILED rc=$RC"; exit $RC; }

echo "=== resolved versions ==="
pip list 2>/dev/null | grep -iE "^(numpy|scipy|pandas|scikit-learn|lightgbm|tabfm|torch|yasa|antropy|neurokit2|edfio) "

echo "=== dependency conflicts ==="
pip check 2>&1 | head -10

echo "=== the Dockerfile weight-bake step ==="
export HF_HOME=/scratch/simran/pn26/reqtest_hf
python -c "
from tabfm import tabfm_v1_0_0_pytorch as hub
m = hub.load('classification', device='cpu')
print('TabFM weights cached OK')
" 2>&1 | tail -5
RC=$?

echo "=== can team_code import under these pins? ==="
PYTHONPATH="$PROJECT" python -c "
import team_code
print('team_code imports, SELECTED_FEATURES =', len(team_code.SELECTED_FEATURES))
" 2>&1 | tail -5

echo "end=$(date -Is) rc=$RC"
exit $RC
