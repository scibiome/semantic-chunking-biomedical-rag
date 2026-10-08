#!/bin/bash
#SBATCH --job-name=s_hrt_ddi_sp
#SBATCH --partition=computedfg
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/s_hrt_ddi_semantic_prop_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/s_hrt_ddi_semantic_prop_%j.err
#SBATCH --mail-type=NONE
#
# S-HRT step for DDI Semantic Prop apply_chunk_scorer output (runs AFTER apply_chunk_scorer_ddi_semantic_prop).
# Uses DDI dev file for relation-to-(head,tail) mapping.
#
# Submit with dependency: sbatch --dependency=afterok:APPLY_JOB_ID slurm/run_s_hrt_ddi_semantic_prop.slurm
# Or standalone after apply completed: sbatch slurm/run_s_hrt_ddi_semantic_prop.slurm

set -euo pipefail

echo "=========================================="
echo "S-HRT: DDI Semantic Prop (apply chunk scorer output)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

DDI_SP_TRAIN="train_triplet_chunk_scorer_ddi_semantic_prop.json"
DDI_SP_TEST="test_triplet_chunk_scorer_ddi_semantic_prop.json"

echo "Checking inputs..."
for f in "$DDI_SP_TRAIN" "$DDI_SP_TEST"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
    if [ ! -s "$f" ]; then
        echo "ERROR: $f is empty"
        exit 1
    fi
done
wc -l "$DDI_SP_TRAIN" "$DDI_SP_TEST"
echo ""

python3 8_make_s_hrt_for_chunk_scorer_improved.py --variant ddi_semantic_prop

EXIT=$?
echo ""
echo "=========================================="
if [ $EXIT -eq 0 ]; then
    echo "S-HRT (ddi_semantic_prop) completed"
    ls -lh train_triplet_chunk_scorer_ddi_semantic_prop_s_hrt.json test_triplet_chunk_scorer_ddi_semantic_prop_s_hrt.json 2>/dev/null || true
else
    echo "S-HRT (ddi_semantic_prop) failed with exit $EXIT"
fi
echo "=========================================="
echo "End time: $(date)"
exit $EXIT
