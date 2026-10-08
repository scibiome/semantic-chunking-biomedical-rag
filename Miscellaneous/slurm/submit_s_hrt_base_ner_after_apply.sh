#!/bin/bash
# Submit S-HRT base_ner to run after apply_chunk_scorer_base_ner completes.
# Usage: ./submit_s_hrt_base_ner_after_apply.sh <APPLY_JOB_ID>
# Example: ./submit_s_hrt_base_ner_after_apply.sh 16385
# Get APPLY_JOB_ID from: squeue -u $USER -n apply_chunk_scorer_base_ner

APPLY_JOB_ID="${1:-}"
if [[ -z "$APPLY_JOB_ID" ]]; then
    echo "Usage: $0 <apply_chunk_scorer_base_ner_job_id>"
    echo "Example: $0 16385"
    echo "List your jobs: squeue -u \$USER"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Submitting s_hrt_base_ner with dependency afterok:$APPLY_JOB_ID"
sbatch --dependency=afterok:"$APPLY_JOB_ID" slurm/run_s_hrt_base_ner.slurm
