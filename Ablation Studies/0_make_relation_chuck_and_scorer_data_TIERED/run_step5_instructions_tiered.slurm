#!/bin/bash
#SBATCH --job-name=tiered_step5
#SBATCH --output=logs/tiered_step5_%j.out
#SBATCH --error=logs/tiered_step5_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=computedfg
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1

set -euo pipefail

# Activate environment
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

# Move to the correct directory
cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_TIERED

# Create logs directory if it doesn't exist
mkdir -p logs

echo "================================================================================"
echo "Step 5: Generate Instruction Files (TIERED Variant)"
echo "================================================================================"
echo "Date: $(date)"
echo "Input: output/train_data_topn_chucks_5_tiered.json"
echo "       output/test_data_topn_chucks_5_tiered.json"
echo "Output: output/train_chuck_instruction_5_tiered.json"
echo "        output/test_chuck_instruction_5_tiered.json"
echo "================================================================================"

python 5_make_chuck_instruction_5_tiered.py

echo ""
echo "================================================================================"
echo "✅ Step 5 Complete!"
echo "Job finished at: $(date)"
echo "================================================================================"
