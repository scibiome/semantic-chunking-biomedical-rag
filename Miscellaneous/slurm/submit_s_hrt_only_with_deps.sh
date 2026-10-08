#!/bin/bash
#
# Submit only S-HRT jobs with dependencies on existing apply_chunk_scorer jobs.
# Use when apply scored / scored_ner have already been submitted.
#
# Usage: ./submit_s_hrt_only_with_deps.sh JOB_APPLY_SCORED JOB_APPLY_SCORED_NER
# Example: ./submit_s_hrt_only_with_deps.sh 15778 15779

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 JOB_APPLY_SCORED JOB_APPLY_SCORED_NER"
    echo "Example: $0 15778 15779"
    exit 1
fi

J_APPLY_SCORED="$1"
J_APPLY_SCORED_NER="$2"
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting S-HRT jobs only (with dependencies)"
echo "=========================================="
echo "  apply_scored job:     $J_APPLY_SCORED"
echo "  apply_scored_ner job: $J_APPLY_SCORED_NER"
echo ""

echo "Submitting s_hrt_scored (dependency: afterok:$J_APPLY_SCORED)..."
J_SHRT_SCORED=$(sbatch --dependency=afterok:$J_APPLY_SCORED run_s_hrt_scored.slurm | awk '{print $4}')
echo "  -> Job ID: $J_SHRT_SCORED"
echo ""

echo "Submitting s_hrt_scored_ner (dependency: afterok:$J_APPLY_SCORED_NER)..."
J_SHRT_SCORED_NER=$(sbatch --dependency=afterok:$J_APPLY_SCORED_NER run_s_hrt_scored_ner.slurm | awk '{print $4}')
echo "  -> Job ID: $J_SHRT_SCORED_NER"
echo ""

echo "=========================================="
echo "S-HRT jobs submitted"
echo "=========================================="
echo "  s_hrt_scored:     $J_SHRT_SCORED (after $J_APPLY_SCORED)"
echo "  s_hrt_scored_ner: $J_SHRT_SCORED_NER (after $J_APPLY_SCORED_NER)"
echo "=========================================="
