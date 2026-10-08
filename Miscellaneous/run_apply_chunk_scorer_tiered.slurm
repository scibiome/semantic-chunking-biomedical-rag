#!/bin/bash
#SBATCH --job-name=apply_scorer_tiered
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dds8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/apply_scorer_tiered_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/apply_scorer_tiered_%j.err

set -euo pipefail

echo "========================================="
echo "Apply Chunk Scorer - Tiered (dds8)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false

cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

CHECKPOINT="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data/1_train_chuck_scorer_model/Our_model_tiered/checkpoint-1000"
TOPN_TRAIN="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_TIERED/output/train_data_topn_chucks_5_tiered.json"
TOPN_TEST="/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_TIERED/output/test_data_topn_chucks_5_tiered.json"
KNN_TRAIN="KNN_demo_train.json"
KNN_TEST="KNN_demo_test.json"
OUT_TRAIN="train_triplet_chunk_scorer_tiered.json"
OUT_TEST="test_triplet_chunk_scorer_tiered.json"

ls -lh "$CHECKPOINT"
ls -lh "$TOPN_TRAIN" "$TOPN_TEST"
ls -lh "$KNN_TRAIN" "$KNN_TEST"

echo "Running apply_chunk_scorer.py for Tiered..."
python3 apply_chunk_scorer.py \
  --checkpoint "$CHECKPOINT" \
  --topn-train "$TOPN_TRAIN" \
  --topn-test "$TOPN_TEST" \
  --knn-train "$KNN_TRAIN" \
  --knn-test "$KNN_TEST" \
  --output-train "$OUT_TRAIN" \
  --output-test "$OUT_TEST" \
  --max-examples 3 \
  --max-length 1024

echo "Completed at: $(date)"
