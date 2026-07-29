#!/bin/bash
# Submit the whole large-dataset pipeline as one dependency chain.
#
# The 6,600-record set matters for two separate reasons. It gives roughly six
# times the age-matched pairs, which should halve confidence intervals that are
# currently +/-0.09 and make methods distinguishable at all. And it repairs a
# structural flaw in the folds: on the small set, holding out S0001 leaves only
# 246 training records, so our one powered fold is also our most data-starved
# one. At 6,600 that fold trains on 1,461 and the inverted fold on 5,139.
#
# Every stage is idempotent, so a failed stage can be rerun without redoing the
# work before it.
#
# Usage:
#   slurm/large_pipeline.sh [dependency_jobid]

set -uo pipefail

PROJECT=/home/simran/sleep-study-cognitive-screening-challenge
cd "$PROJECT" || exit 1

DATA=/scratch/simran/pn26/raw/training_set_large
PARTS=/scratch/simran/pn26/processed/large_v7_parts
PS_OUT=/scratch/simran/pn26/processed/ps_large
FEATS=data/processed/features_large_v7.pkl
FEATS_V8=data/processed/features_large_v8.pkl

BAD=$(cat /tmp/pn26_bad_nodes.txt 2>/dev/null || echo "")
EXCLUDE=${BAD:+--exclude=$BAD}
WAIT_FOR="${1:-}"
DEP=${WAIT_FOR:+--dependency=afterok:$WAIT_FOR}

mkdir -p "$PARTS" "$PS_OUT" slurm/logs

if [ ! -d "$DATA/physiological_data" ] && [ -z "$WAIT_FOR" ]; then
    echo "ERROR: $DATA is not populated and no dependency job was given." >&2
    echo "Pass the download job id, e.g. slurm/large_pipeline.sh 2121336" >&2
    exit 1
fi

echo "data=$DATA"
echo "waiting on=${WAIT_FOR:-nothing}"

# 1. Hand-crafted features. 64 tasks keeps us inside the 100-job QOS cap on
#    short; quick allows only 20 and 64 CPUs, so it cannot host an array.
J_EXTRACT=$(sbatch --parsable --job-name=pn26L_extract --partition=short \
    --array=0-63%32 $EXCLUDE $DEP \
    --export=ALL,DATA_FOLDER="$DATA",OUTDIR="$PARTS" \
    slurm/extract.sh)
echo "extract      = $J_EXTRACT"

J_MERGE=$(sbatch --parsable --job-name=pn26L_merge --partition=short \
    $EXCLUDE --dependency=afterany:$J_EXTRACT \
    --cpus-per-task=8 --mem=64G --time=02:00:00 slurm/run.sh \
    python scripts/merge_features.py --parts "$PARTS" \
        --demographics "$DATA/demographics.csv" --out "$FEATS")
echo "merge        = $J_MERGE"

J_SELF=$(sbatch --parsable --job-name=pn26L_selfnorm --partition=short \
    $EXCLUDE --dependency=afterok:$J_MERGE \
    --cpus-per-task=8 --mem=64G --time=02:00:00 slurm/run.sh \
    python src/data/self_norm.py --features "$FEATS" --out "$FEATS_V8")
echo "self_norm    = $J_SELF"

J_BASE=$(sbatch --parsable --job-name=pn26L_baselines --partition=short \
    $EXCLUDE --dependency=afterok:$J_SELF \
    --cpus-per-task=16 --mem=96G --time=12:00:00 slurm/run.sh \
    python scripts/run_baselines.py --features "$FEATS_V8" \
        --out results/baselines_large_v8.json)
echo "baselines    = $J_BASE"

J_TUNE=$(sbatch --parsable --job-name=pn26L_tune --partition=short \
    $EXCLUDE --dependency=afterok:$J_SELF \
    --cpus-per-task=16 --mem=96G --time=12:00:00 slurm/run.sh \
    python scripts/tune_lgbm.py --features "$FEATS_V8" \
        --out results/lgbm_tuning_large.json)
echo "tune         = $J_TUNE"

# 2. Philosopher's Stone latents. Independent of the hand-feature branch, so it
#    only waits on the data itself. The wavelet spectrogram is ~65-90s per
#    record on CPU while the forward pass is ~0.1s on GPU, so 12 shards over
#    6,600 records is roughly 10 hours.
J_PS=$(sbatch --parsable --job-name=pn26L_ps --array=0-11%12 $EXCLUDE $DEP \
    --export=ALL,DATA_FOLDER="$DATA",OUTDIR="$PS_OUT" \
    slurm/ps_extract.sh)
echo "ps_extract   = $J_PS"

J_PSEVAL=$(sbatch --parsable --job-name=pn26L_pseval --partition=short \
    $EXCLUDE --dependency=afterany:$J_PS,afterok:$J_SELF \
    --cpus-per-task=16 --mem=96G --time=12:00:00 slurm/run.sh \
    python scripts/eval_ps_latents.py --ps-dir "$PS_OUT" \
        --features "$FEATS_V8" --out results/ps_eval_large.json)
echo "ps_eval      = $J_PSEVAL"

echo
echo "chain submitted. watch with: squeue -u $USER"
