#!/bin/bash
#SBATCH --job-name=regen_train_data
#SBATCH --output=logs/regen_train_%j.out
#SBATCH --error=logs/regen_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=debug

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

echo "1. Regenerating Fixed Instructions (Step 5)..."
cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data
python 5_make_chuck_instruction_5.py

echo "2. Merging Fixed Train Data..."
cd ../2_relation_data_to_triple_train_data
python chuck_triplet_progress_train_fixed_FIXED.py

echo "3. Merging Improved Train Data..."
# Note: Improved Step 5 was already run, but we need to merge train data
python chuck_triplet_progress_train_improved_FIXED.py

echo "Done. Training files are clean."


