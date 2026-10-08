#!/bin/bash
# Submit apply_chunk_scorer_base_ner to run after chunk_scorer_base_ner training completes.
# Usage: ./submit_apply_base_ner_after_training.sh <TRAIN_JOB_ID>
# Example: ./submit_apply_base_ner_after_training.sh 16381
# Get TRAIN_JOB_ID from: squeue -u $USER -n chunk_scorer_base_ner

TRAIN_JOB_ID="${1:-}"
if [[ -z "$TRAIN_JOB_ID" ]]; then
  echo "Usage: $0 <chunk_scorer_base_ner_training_job_id>"
  echo "Example: $0 16381"
  echo "List your jobs: squeue -u \$USER"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Submitting apply_chunk_scorer_base_ner with dependency afterok:$TRAIN_JOB_ID"
sbatch --dependency=afterok:"$TRAIN_JOB_ID" slurm/apply_chunk_scorer_base_ner.slurm
