#!/bin/bash
#SBATCH --job-name=tiered_step3
#SBATCH --output=logs/tiered_step3_%j.out
#SBATCH --error=logs/tiered_step3_%j.err
#SBATCH --time=04:00:00
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

echo "================================================================================"
echo "Step 3: Make Top-N Chunks for TRAIN Data (TIERED Variant)"
echo "================================================================================"
echo "Date: $(date)"
echo "Config: configs/config.json (Tiered Triggers: Enabled, Hierarchy: Disabled)"
echo "Input: output/train_data_chucks_5_tiered.json"
echo "Output: output/train_data_topn_chucks_5_tiered.json"
echo "================================================================================"

python 3_make_topn_for_train_tiered.py

echo ""
echo "================================================================================"
echo "✅ Step 3 Complete!"
echo "Job finished at: $(date)"
echo "================================================================================"
