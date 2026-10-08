#!/bin/bash
#SBATCH --job-name=tiered_step1b
#SBATCH --output=logs/tiered_step1b_%j.out
#SBATCH --error=logs/tiered_step1b_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --nodelist=dds8

set -euo pipefail

# Activate environment
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Move to the correct directory
cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_TIERED

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Step 1b: Generate embeddings for stored chunks (TIERED variant)"
echo "Date: $(date)"
echo "Input: output/stored_chucks_with_relation_5_tiered.txt"
echo "Output: output/retrived_chucks_5_tiered.npy"

python 1b_get_store_chuck_embeddings_tiered.py

echo "Job finished at: $(date)"
