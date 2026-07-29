#!/bin/bash
# Submit 3 parallel GPU jobs (one per LOSO fold) + 1 CPU combine job.
# Usage: bash slurm/submit_tabpfn_selection.sh

REPO="/home/simran/sleep-study-cognitive-screening-challenge"
DATA="${REPO}/data/processed/features_training_v3.pkl"
OUTDIR="${REPO}/results/tabpfn_selection"
LOGDIR="${REPO}/slurm/logs"
SCRIPTS="${REPO}/scripts/tabpfn"

mkdir -p "${OUTDIR}" "${LOGDIR}"

SITES=("S0001" "I0002" "I0006")
FOLD_JIDS=""

for SITE in "${SITES[@]}"; do
    JID=$(sbatch \
        --job-name="tabpfn_${SITE}" \
        --partition=short \
        --time=01:00:00 \
        --mem=32G \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --exclude=gpu-6-[01-20] \
        --output="${LOGDIR}/tabpfn_${SITE}_%j.out" \
        --error="${LOGDIR}/tabpfn_${SITE}_%j.err" \
        --wrap="
source ~/.physionet2026_env.sh 2>/dev/null
eval \"\$(/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/condabin/conda shell.bash hook)\"
conda activate pn26
cd ${REPO}
python ${SCRIPTS}/run_fold.py --fold-site ${SITE} --data ${DATA} --outdir ${OUTDIR} --n-repeats 10
" | awk '{print $4}')

    echo "Submitted fold ${SITE}: job ${JID}"
    if [ -z "${FOLD_JIDS}" ]; then
        FOLD_JIDS="${JID}"
    else
        FOLD_JIDS="${FOLD_JIDS}:${JID}"
    fi
done

# Combine job — runs after all 3 folds complete
COMBINE_JID=$(sbatch \
    --job-name="tabpfn_combine" \
    --partition=short \
    --time=00:10:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --dependency=afterok:${FOLD_JIDS} \
    --output="${LOGDIR}/tabpfn_combine_%j.out" \
    --error="${LOGDIR}/tabpfn_combine_%j.err" \
    --wrap="
source ~/.physionet2026_env.sh 2>/dev/null
eval \"\$(/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/condabin/conda shell.bash hook)\"
conda activate pn26
cd ${REPO}
python ${SCRIPTS}/combine_folds.py --indir ${OUTDIR} --outdir ${OUTDIR}
" | awk '{print $4}')

echo "Submitted combine: job ${COMBINE_JID} (after ${FOLD_JIDS})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: ${OUTDIR}/"
