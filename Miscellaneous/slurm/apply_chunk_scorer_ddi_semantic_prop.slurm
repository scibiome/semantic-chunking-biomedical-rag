#!/bin/bash
#SBATCH --job-name=apply_chunk_ddi_sp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dds9
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/apply_chunk_scorer_ddi_semantic_prop_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/apply_chunk_scorer_ddi_semantic_prop_%j.err
#SBATCH --mail-type=NONE

# Apply trained chunk scorer (DDI Semantic+Proposition) to re-rank top-N chunks and produce triplet JSONL.
# Requires: DDI Semantic Prop chunk scorer job finished → Our_model_DDI_SemanticProp/checkpoint-1000

echo "=========================================="
echo "Apply Chunk Scorer - DDI Semantic+Proposition"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

CHECKPOINT="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data/1_train_chuck_scorer_model/Our_model_DDI_SemanticProp/checkpoint-1000"
DDI_SP_OUT="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_DDI_IMPROVED/DDI_SemanticProp/output"

echo "Configuration (--variant ddi_semantic_prop):"
echo "  Checkpoint: $CHECKPOINT"
echo "  TopN: $DDI_SP_OUT/train_data_topn_chucks_5_semantic_prop_ddi.json (train/test)"
echo "  KNN: ./KNN_ddi_train.json, ./KNN_ddi_test.json"
echo "  Output: train_triplet_chunk_scorer_ddi_semantic_prop.json, test_triplet_chunk_scorer_ddi_semantic_prop.json"
echo ""

echo "Verifying inputs..."
if [ ! -d "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT (run DDI Semantic Prop chunk scorer training first)"
    exit 1
fi
for f in "$DDI_SP_OUT/train_data_topn_chucks_5_semantic_prop_ddi.json" "$DDI_SP_OUT/test_data_topn_chucks_5_semantic_prop_ddi.json" "KNN_ddi_train.json" "KNN_ddi_test.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done
echo "✓ All inputs found"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU (before):"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    echo ""
fi

echo "Running apply_chunk_scorer.py --variant ddi_semantic_prop ..."
python3 apply_chunk_scorer.py --variant ddi_semantic_prop --max-examples 3 --max-length 4096

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Apply Chunk Scorer (DDI Semantic+Proposition) COMPLETED"
else
    echo "❌ Apply Chunk Scorer FAILED (exit $EXIT_CODE)"
fi
echo "=========================================="
ls -lh train_triplet_chunk_scorer_ddi_semantic_prop.json test_triplet_chunk_scorer_ddi_semantic_prop.json 2>/dev/null || true
echo "Completed at: $(date)"
exit $EXIT_CODE
