#!/bin/bash
#SBATCH --job-name=s_hrt_scored
#SBATCH --partition=computedfg
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/s_hrt_scored_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/s_hrt_scored_%j.err
#SBATCH --mail-type=NONE
#
# S-HRT step for scored apply_chunk_scorer output.
# Submit with: sbatch --dependency=afterok:JOBID run_s_hrt_scored.slurm
# where JOBID is the apply_chunk_scorer_scored job.

set -euo pipefail

echo "=========================================="
echo "S-HRT: Scored"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

SCORED_TRAIN="train_triplet_chunk_scorer_scored.json"
SCORED_TEST="test_triplet_chunk_scorer_scored.json"

echo "Checking inputs..."
for f in "$SCORED_TRAIN" "$SCORED_TEST"; do
    if [ ! -f "$f" ]; then echo "ERROR: Missing $f"; exit 1; fi
    if [ ! -s "$f" ]; then echo "ERROR: $f is empty"; exit 1; fi
done
wc -l "$SCORED_TRAIN" "$SCORED_TEST"
echo ""

python3 8_make_s_hrt_for_chunk_scorer_improved.py \
    --train-input "$SCORED_TRAIN" \
    --test-input "$SCORED_TEST" \
    --dev-source "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"

EXIT=$?
echo ""
echo "=========================================="
if [ $EXIT -eq 0 ]; then
    echo "S-HRT (scored) completed"
    ls -lh train_triplet_chunk_scorer_scored_s_hrt*.json test_triplet_chunk_scorer_scored_s_hrt*.json 2>/dev/null || true
else
    echo "S-HRT (scored) failed with exit $EXIT"
fi
echo "=========================================="
echo "End time: $(date)"
exit $EXIT
