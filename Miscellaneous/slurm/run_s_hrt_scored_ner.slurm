#!/bin/bash
#SBATCH --job-name=s_hrt_scored_ner
#SBATCH --partition=computedfg
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/s_hrt_scored_ner_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/s_hrt_scored_ner_%j.err
#SBATCH --mail-type=NONE
#
# S-HRT step for scored_ner apply_chunk_scorer output.
# Submit with: sbatch --dependency=afterok:JOBID run_s_hrt_scored_ner.slurm
# where JOBID is the apply_chunk_scorer_scored_ner job.

set -euo pipefail

echo "=========================================="
echo "S-HRT: Scored NER"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

SCORED_NER_TRAIN="train_triplet_chunk_scorer_scored_ner.json"
SCORED_NER_TEST="test_triplet_chunk_scorer_scored_ner.json"

echo "Checking inputs..."
for f in "$SCORED_NER_TRAIN" "$SCORED_NER_TEST"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
    if [ ! -s "$f" ]; then
        echo "ERROR: $f is empty"
        exit 1
    fi
done
wc -l "$SCORED_NER_TRAIN" "$SCORED_NER_TEST"
echo ""

python3 8_make_s_hrt_for_chunk_scorer_improved.py \
    --train-input "$SCORED_NER_TRAIN" \
    --test-input "$SCORED_NER_TEST" \
    --dev-source "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"

EXIT=$?
echo ""
echo "=========================================="
if [ $EXIT -eq 0 ]; then
    echo "S-HRT (scored_ner) completed"
    ls -lh train_triplet_chunk_scorer_scored_ner_s_hrt.json train_triplet_chunk_scorer_scored_ner_s_hrt3.json \
           test_triplet_chunk_scorer_scored_ner_s_hrt.json test_triplet_chunk_scorer_scored_ner_s_hrt3.json 2>/dev/null || true
else
    echo "S-HRT (scored_ner) failed with exit $EXIT"
fi
echo "=========================================="
echo "End time: $(date)"
exit $EXIT
