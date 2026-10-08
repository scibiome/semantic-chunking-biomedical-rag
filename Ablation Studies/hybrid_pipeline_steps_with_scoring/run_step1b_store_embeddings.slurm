#!/bin/bash
#SBATCH --job-name=step1b_scoring_emb
#SBATCH --output=/home/users/riah/project_thesis/logs/step1b_scoring_emb_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/step1b_scoring_emb_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=dds9

set -euo pipefail
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false

cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring

echo "--- STEP 1b: Store Chunk Embeddings (SCORING PIPELINE) ---"
echo "Input: hybrid_gm-ciht_relation_with_chunks_5_with_scoring.json"
python3 -u hybrid_1b_store_embeddings.py
echo "✅ Step 1b Completed."
