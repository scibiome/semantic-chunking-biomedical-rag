#!/bin/bash
#
# Submit only trainer jobs (scored + scored_ner S-HRT 1-example) with dependencies on S-HRT jobs.
# Use when apply + S-HRT have already been run and *_s_hrt.json files exist.
#
# Usage: ./submit_trainer_only_with_deps.sh JOB_SHRT_SCORED JOB_SHRT_SCORED_NER
# Example: ./submit_trainer_only_with_deps.sh 15780 15781

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 JOB_SHRT_SCORED JOB_SHRT_SCORED_NER"
    echo "  JOB_SHRT_SCORED     = s_hrt_scored job ID"
    echo "  JOB_SHRT_SCORED_NER = s_hrt_scored_ner job ID"
    echo "Example: $0 15780 15781"
    exit 1
fi

J_SHRT_SCORED="$1"
J_SHRT_SCORED_NER="$2"
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIOMEDRAG="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER_SLURM="$BIOMEDRAG/3_trainning_triple_model"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting trainer jobs only (S-HRT 1-example, with dependencies)"
echo "=========================================="
echo "  s_hrt_scored job:     $J_SHRT_SCORED"
echo "  s_hrt_scored_ner job: $J_SHRT_SCORED_NER"
echo ""

echo "Submitting train_scored_s_hrt_1024 (dependency: afterok:$J_SHRT_SCORED)..."
J_TRAIN_SCORED=$(sbatch --dependency=afterok:$J_SHRT_SCORED "$TRAINER_SLURM/slurm_train_scored_s_hrt_1024_dds8.slurm" | awk '{print $4}')
echo "  -> Job ID: $J_TRAIN_SCORED"
echo ""

echo "Submitting train_scored_ner_s_hrt_1024 (dependency: afterok:$J_SHRT_SCORED_NER)..."
J_TRAIN_SCORED_NER=$(sbatch --dependency=afterok:$J_SHRT_SCORED_NER "$TRAINER_SLURM/slurm_train_scored_ner_s_hrt_1024_dds9.slurm" | awk '{print $4}')
echo "  -> Job ID: $J_TRAIN_SCORED_NER"
echo ""

echo "=========================================="
echo "Trainer jobs submitted"
echo "=========================================="
echo "  train_scored_s_hrt_1024:     $J_TRAIN_SCORED (after $J_SHRT_SCORED, dds8)"
echo "  train_scored_ner_s_hrt_1024: $J_TRAIN_SCORED_NER (after $J_SHRT_SCORED_NER, dds9)"
echo "  Inputs: *_s_hrt.json (S-HRT 1-example)"
echo "=========================================="
