#!/bin/bash
#SBATCH --job-name=tiered_step2b
#SBATCH --output=logs/tiered_step2b_%j.out
#SBATCH --error=logs/tiered_step2b_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
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

echo "Starting Step 2b: Create similar chunks for TEST data (TIERED variant)"
echo "Date: $(date)"
echo "Config: configs/config.json (Tiered Triggers: Enabled, Hierarchy: Disabled)"
echo "Input: ../dataset/0_GM-CIHT/renew_triplet_T_test_deduplicated.jsonl"
echo "Output: output/test_data_chucks_5_tiered.json"

python 2_make_test_chunks_5_tiered.py

echo "Job finished at: $(date)"
