#!/bin/bash
#SBATCH --job-name=sem_prop_part2
#SBATCH --output=logs/sem_prop_part2_%j.out
#SBATCH --error=logs/sem_prop_part2_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1

set -euo pipefail

# Activate environment
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Move to directory
cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_SEMANTIC_PROP

echo "================================================================================"
echo "🚀 Running Semantic + Proposition Pipeline Part 2 (Steps 2b - 5b)"
echo "================================================================================"
echo "Date: $(date)"
echo ""

echo "--------------------------------------------------------------------------------"
echo "Step 2b: Create similar chunks for TEST data"
echo "--------------------------------------------------------------------------------"
python 2_make_test_chunks_5_semantic_prop.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 3: Make Top-N chunks for TRAIN"
echo "--------------------------------------------------------------------------------"
python 3_make_topn_for_train_semantic_prop.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 3b: Make Top-N chunks for TEST"
echo "--------------------------------------------------------------------------------"
python 3_make_topn_for_test_semantic_prop.py

echo ""
echo "================================================================================"
echo "✅ Pipeline Part 2 Complete (Stopped before Step 4 - embeddings will use different script)"
echo "Date: $(date)"
echo "================================================================================"
echo ""
echo "NOTE: Steps 4, 5, and 5b will be run separately with a different embedding script."
