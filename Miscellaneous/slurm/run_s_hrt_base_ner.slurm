#!/bin/bash
#SBATCH --job-name=s_hrt_base_ner
#SBATCH --partition=computedfg
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/s_hrt_base_ner_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/s_hrt_base_ner_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.ahuja@tu-braunschweig.de
#
# S-HRT step for base_ner apply_chunk_scorer output (runs AFTER apply_chunk_scorer_base_ner).
#
# Submit with dependency on apply_chunk_scorer_base_ner job:
#   sbatch --dependency=afterok:<APPLY_JOB_ID> slurm/run_s_hrt_base_ner.slurm
# Or use: ./submit_s_hrt_base_ner_after_apply.sh <APPLY_JOB_ID>

set -euo pipefail

echo "=========================================="
echo "S-HRT: Base NER (apply chunk scorer output)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

BASE_NER_TRAIN="train_triplet_chunk_scorer_base_ner.json"
BASE_NER_TEST="test_triplet_chunk_scorer_base_ner.json"

echo "Checking inputs..."
for f in "$BASE_NER_TRAIN" "$BASE_NER_TEST"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
    if [ ! -s "$f" ]; then
        echo "ERROR: $f is empty"
        exit 1
    fi
done
wc -l "$BASE_NER_TRAIN" "$BASE_NER_TEST"
echo ""

python3 8_make_s_hrt_for_chunk_scorer_improved.py --variant base_ner

EXIT=$?
echo ""
echo "=========================================="
if [ $EXIT -eq 0 ]; then
    echo "S-HRT (base_ner) completed"
    ls -lh train_triplet_chunk_scorer_base_ner_s_hrt.json train_triplet_chunk_scorer_base_ner_s_hrt3.json \
           test_triplet_chunk_scorer_base_ner_s_hrt.json test_triplet_chunk_scorer_base_ner_s_hrt3.json 2>/dev/null || true
else
    echo "S-HRT (base_ner) failed with exit $EXIT"
fi
echo "=========================================="
echo "End time: $(date)"
exit $EXIT
