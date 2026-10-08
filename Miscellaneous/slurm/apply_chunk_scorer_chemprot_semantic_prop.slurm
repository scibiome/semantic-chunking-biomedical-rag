#!/bin/bash
#SBATCH --job-name=apply_chunk_chemprot_sp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dds9
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/apply_chunk_scorer_chemprot_semantic_prop_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/apply_chunk_scorer_chemprot_semantic_prop_%j.err
#SBATCH --mail-type=NONE

# Apply trained chunk scorer (ChemProt Semantic+Proposition) to re-rank top-N chunks and produce triplet JSONL.
# Requires: ChemProt Semantic Prop chunk scorer job finished → Our_model_Chemprot_SemanticProp/checkpoint-1000

echo "=========================================="
echo "Apply Chunk Scorer - ChemProt Semantic+Proposition"
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

CHECKPOINT="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data/1_train_chuck_scorer_model/Our_model_Chemprot_SemanticProp/checkpoint-1000"
CHEMPROT_SP_OUT="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_CHEMPROT_IMPROVED/Chemprot_SemanticProp/output"

echo "Configuration (--variant chemprot_semantic_prop):"
echo "  Checkpoint: $CHECKPOINT"
echo "  TopN: $CHEMPROT_SP_OUT/train_data_topn_chucks_5_semantic_prop_chemprot.json (train/test)"
echo "  KNN: ./KNN_chemprot_train.json, ./KNN_chemprot_test.json"
echo "  Output: train_triplet_chunk_scorer_chemprot_semantic_prop.json, test_triplet_chunk_scorer_chemprot_semantic_prop.json"
echo ""

echo "Verifying inputs..."
if [ ! -d "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT (run ChemProt Semantic Prop chunk scorer training first)"
    exit 1
fi
for f in "$CHEMPROT_SP_OUT/train_data_topn_chucks_5_semantic_prop_chemprot.json" "$CHEMPROT_SP_OUT/test_data_topn_chucks_5_semantic_prop_chemprot.json" "KNN_chemprot_train.json" "KNN_chemprot_test.json"; do
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

echo "Running apply_chunk_scorer.py --variant chemprot_semantic_prop ..."
python3 apply_chunk_scorer.py --variant chemprot_semantic_prop --max-examples 3 --max-length 1024

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Apply Chunk Scorer (ChemProt Semantic+Proposition) COMPLETED"
else
    echo "❌ Apply Chunk Scorer FAILED (exit $EXIT_CODE)"
fi
echo "=========================================="
ls -lh train_triplet_chunk_scorer_chemprot_semantic_prop.json test_triplet_chunk_scorer_chemprot_semantic_prop.json 2>/dev/null || true
echo "Completed at: $(date)"
exit $EXIT_CODE
