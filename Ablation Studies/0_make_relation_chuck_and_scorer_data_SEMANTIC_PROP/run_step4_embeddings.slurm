#!/bin/bash
#SBATCH --job-name=sem_prop_step4
#SBATCH --output=logs/sem_prop_step4_%j.out
#SBATCH --error=logs/sem_prop_step4_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --nodelist=dds9

set -euo pipefail

# Activate environment
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Move to the correct directory
cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_SEMANTIC_PROP

# Create logs directory if it doesn't exist
mkdir -p logs

echo "================================================================================"
echo "Step 4: Generate Candidate Embeddings (Semantic + Proposition Variant)"
echo "================================================================================"
echo "Date: $(date)"
echo "Input: output/train_data_topn_chucks_5_semantic_prop.json"
echo "       output/test_data_topn_chucks_5_semantic_prop.json"
echo "Output: output/embeddings_shards/ (shard files)"
echo "        output/all_candiate_chunks_5_semantic_prop.txt"
echo "Model: /home/users/riah/models/MedLLaMA_13B"
echo "================================================================================"

python 4_make_candiate_chucks_embeddings_semantic_prop.py

echo ""
echo "================================================================================"
echo "✅ Step 4 Complete!"
echo "Job finished at: $(date)"
echo "================================================================================"
