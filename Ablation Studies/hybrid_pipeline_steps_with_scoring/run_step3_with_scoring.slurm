#!/bin/bash
#SBATCH --job-name=step3_scoring_topn
#SBATCH --output=/home/users/riah/project_thesis/logs/step3_scoring_topn_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/step3_scoring_topn_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false

cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring

echo "--- STEP 3: Extract TopN Similar Chunks (SCORING PIPELINE) ---"
echo "Input: hybrid_train/test_data_chucks_5_with_scoring.json"
echo "Output: hybrid_train/test_data_topn_scored_5_with_scoring.json"
python3 -u hybrid_3_make_topn_scored_with_scoring.py
echo "✅ Step 3 Completed."
