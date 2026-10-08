#!/bin/bash
#SBATCH --job-name=improved_triplet_test
#SBATCH --output=/home/users/riah/project_thesis/logs/improved_triplet_test_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/improved_triplet_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

WORKTREE=/home/users/riah/.cursor/worktrees/BIoMedRAG__SSH__134.169.129.163_/ka1a9
STEP_DIR=$WORKTREE/2_relation_data_to_triple_train_data

echo "================================================================================"
echo "🚀 IMPROVED TRIPLET TEST (deduplicated)"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Time: $(date)"
echo "================================================================================"

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false

cd "$STEP_DIR"

python chuck_triplet_progress_test_improved.py

echo "================================================================================"
echo "✅ Improved triplet test data regenerated (deduplicated)"
echo "================================================================================"
echo "Generated file: test_chuck_5_triplet_llama2_13b_right_0_improved_deduplicated.json"
echo "Lines: $(wc -l test_chuck_5_triplet_llama2_13b_right_0_improved_deduplicated.json | cut -d' ' -f1)"
echo "Completed at: $(date)"
