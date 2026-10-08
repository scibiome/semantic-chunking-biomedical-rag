#!/bin/bash
#SBATCH --job-name=step1_scoring_ner
#SBATCH --output=/home/users/riah/project_thesis/logs/step1_scoring_ner_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/step1_scoring_ner_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=dds8

set -euo pipefail
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false

cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring

echo "--- STEP 1: Relation-Chunk Mapping (WITH TIERED SCORING + NER) ---"
echo "Variant: Combines cosine similarity (70%) with tiered trigger/hierarchy scores (30%)"
echo "Entity Detection: Pre-trained NER system (biomedical-ner-all)"
python3 -u hybrid_1_chunk_mapping_with_scoring_ner.py
echo "✅ Step 1 Completed."
