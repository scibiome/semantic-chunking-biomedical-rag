#!/bin/bash
#SBATCH --job-name=sem_prop_pipeline
#SBATCH --output=logs/sem_prop_pipeline_%j.out
#SBATCH --error=logs/sem_prop_pipeline_%j.err
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
cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_TIERED

echo "================================================================================"
echo "🚀 Running Semantic + Proposition Pipeline (Steps 1b - 5)"
echo "================================================================================"
echo "Date: $(date)"
echo ""

echo "--------------------------------------------------------------------------------"
echo "Step 1b: Generate embeddings for stored chunks"
echo "--------------------------------------------------------------------------------"
python 1b_get_store_chuck_embeddings_tiered.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 2: Create similar chunks for TRAIN data"
echo "--------------------------------------------------------------------------------"
python 2_make_similar_chuck_for_train_tiered.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 2b: Create similar chunks for TEST data"
echo "--------------------------------------------------------------------------------"
python 2_make_test_chunks_5_tiered.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 3: Make Top-N chunks for TRAIN"
echo "--------------------------------------------------------------------------------"
python 3_make_topn_for_train_tiered.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 3b: Make Top-N chunks for TEST"
echo "--------------------------------------------------------------------------------"
python 3_make_topn_for_test_tiered.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 4: Create candidate embeddings (TRAIN only - for Scorer training if needed)"
echo "--------------------------------------------------------------------------------"
python 4_make_candiate_chucks_embeddings_tiered.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 5: Create instruction files for TRAIN"
echo "--------------------------------------------------------------------------------"
python 5_make_chuck_instruction_5_tiered.py

echo ""
echo "--------------------------------------------------------------------------------"
echo "Step 5b: Create instruction files for TEST"
echo "--------------------------------------------------------------------------------"
python 5_make_chuck_instruction_test_5_tiered.py

echo ""
echo "================================================================================"
echo "✅ Pipeline Complete!"
echo "Date: $(date)"
echo "================================================================================"
