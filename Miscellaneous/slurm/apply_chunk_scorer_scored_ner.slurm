#!/bin/bash
#SBATCH --job-name=apply_chunk_scorer_scored_ner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dds9
#SBATCH --mem=80G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/apply_chunk_scorer_scored_ner_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/apply_chunk_scorer_scored_ner_%j.err
#SBATCH --mail-type=NONE

echo "=========================================="
echo "Apply Chunk Scorer - Scored NER (hybrid pipeline)"
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

CHECKPOINT="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data/1_train_chuck_scorer_model/Our_model_scored_ner/checkpoint-1000"
TOPN_TRAIN="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_train_data_topn_scored_5_with_scoring_ner.json"
TOPN_TEST="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_test_data_topn_scored_5_with_scoring_ner.json"
KNN_TRAIN="./KNN_demo_train.json"
KNN_TEST="./KNN_demo_test.json"
OUTPUT_TRAIN="./train_triplet_chunk_scorer_scored_ner.json"
OUTPUT_TEST="./test_triplet_chunk_scorer_scored_ner.json"

echo "Configuration (scored_ner):"
echo "  Checkpoint: $CHECKPOINT"
echo "  TopN Train: $TOPN_TRAIN"
echo "  TopN Test: $TOPN_TEST"
echo "  KNN Train: $KNN_TRAIN"
echo "  KNN Test: $KNN_TEST"
echo "  Output Train: $OUTPUT_TRAIN"
echo "  Output Test: $OUTPUT_TEST"
echo ""

echo "Verifying input files..."
for f in "$CHECKPOINT/adapter_config.json" "$TOPN_TRAIN" "$TOPN_TEST" "$KNN_TRAIN" "$KNN_TEST"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Not found: $f"
        exit 1
    fi
done
echo "✓ All input files found"

echo "Verifying KNN vs TopN record counts (must match for alignment)..."
python3 -c "
import json
def cnt(p):
    n=0
    with open(p) as f:
        for L in f:
            if L.strip():
                try: json.loads(L); n+=1
                except: pass
    return n
kt,kv=cnt('$KNN_TRAIN'),cnt('$KNN_TEST')
tt,tv=cnt('$TOPN_TRAIN'),cnt('$TOPN_TEST')
ok=(kt==tt and kv==tv)
print(f'  KNN train={kt}  TopN train={tt}  match={kt==tt}')
print(f'  KNN test={kv}   TopN test={tv}   match={kv==tv}')
if not ok:
    print('ERROR: KNN and TopN counts must match')
    exit(1)
print('  KNN/TopN alignment OK')
"
[ $? -ne 0 ] && exit 1
wc -l "$TOPN_TRAIN" "$TOPN_TEST" "$KNN_TRAIN" "$KNN_TEST"
echo ""

echo "Starting apply chunk scorer (scored_ner)..."
python3 apply_chunk_scorer.py --variant scored_ner --max-examples 3 --max-length 1024

EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Apply Chunk Scorer (scored_ner) COMPLETED"
else
    echo "❌ Apply Chunk Scorer (scored_ner) FAILED with exit code $EXIT_CODE"
fi
echo "=========================================="
ls -lh "$OUTPUT_TRAIN" "$OUTPUT_TEST" 2>/dev/null || true
wc -l "$OUTPUT_TRAIN" "$OUTPUT_TEST" 2>/dev/null || true
echo "Completed at: $(date)"
exit $EXIT_CODE
