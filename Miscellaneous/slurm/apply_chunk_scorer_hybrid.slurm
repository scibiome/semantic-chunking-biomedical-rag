#!/bin/bash
#SBATCH --job-name=apply_chunk_scorer_hybrid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/apply_chunk_scorer_hybrid_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/apply_chunk_scorer_hybrid_%j.err
#SBATCH --mail-type=NONE

echo "=========================================="
echo "Apply Chunk Scorer - Hybrid Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate environment
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

# Set paths - using absolute path to hybrid checkpoint
CHECKPOINT="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data/1_train_chuck_scorer_model/Our_model_hybrid_chunk_scorer_5_hybrid/checkpoint-1000"
TOPN_TRAIN="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps/hybrid_train_data_topn_scored_5.json"
TOPN_TEST="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps/hybrid_test_data_topn_scored_5.json"
KNN_TRAIN="./KNN_demo_train.json"
KNN_TEST="./KNN_demo_test.json"
OUTPUT_TRAIN="./train_triplet_chunk_scorer_hybrid.json"
OUTPUT_TEST="./test_triplet_chunk_scorer_hybrid.json"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  TopN Train: $TOPN_TRAIN"
echo "  TopN Test: $TOPN_TEST"
echo "  KNN Train: $KNN_TRAIN"
echo "  KNN Test: $KNN_TEST"
echo "  Output Train: $OUTPUT_TRAIN"
echo "  Output Test: $OUTPUT_TEST"
echo ""

# Verify input files exist
echo "Verifying input files..."
if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Expected adapter_config.json in checkpoint directory"
    exit 1
fi
if [ ! -f "$TOPN_TRAIN" ]; then
    echo "ERROR: TopN train file not found at $TOPN_TRAIN"
    exit 1
fi
if [ ! -f "$TOPN_TEST" ]; then
    echo "ERROR: TopN test file not found at $TOPN_TEST"
    exit 1
fi
if [ ! -f "$KNN_TRAIN" ]; then
    echo "ERROR: KNN train file not found at $KNN_TRAIN"
    exit 1
fi
if [ ! -f "$KNN_TEST" ]; then
    echo "ERROR: KNN test file not found at $KNN_TEST"
    exit 1
fi
echo "✓ All input files found"
echo ""

# Check line counts
echo "Input file line counts:"
wc -l "$TOPN_TRAIN" "$TOPN_TEST" "$KNN_TRAIN" "$KNN_TEST"
echo ""

# Run chunk scorer application
echo "Starting chunk scorer application with hybrid model..."
python3 apply_chunk_scorer.py \
    --checkpoint "$CHECKPOINT" \
    --topn-train "$TOPN_TRAIN" \
    --topn-test "$TOPN_TEST" \
    --knn-train "$KNN_TRAIN" \
    --knn-test "$KNN_TEST" \
    --output-train "$OUTPUT_TRAIN" \
    --output-test "$OUTPUT_TEST" \
    --max-examples 3 \
    --max-length 1024

echo ""
echo "=========================================="
echo "Apply Chunk Scorer COMPLETED"
echo "=========================================="
echo "Output files:"
ls -lh "$OUTPUT_TRAIN" "$OUTPUT_TEST" 2>/dev/null || echo "Output files not found"
echo ""
echo "Line counts:"
wc -l "$OUTPUT_TRAIN" "$OUTPUT_TEST" 2>/dev/null || echo "Could not count lines"
echo ""
echo "Completed at: $(date)"

