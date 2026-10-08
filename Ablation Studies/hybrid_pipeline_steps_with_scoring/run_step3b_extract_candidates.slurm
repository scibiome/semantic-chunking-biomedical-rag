#!/bin/bash
#SBATCH --job-name=step3b_scoring_extract
#SBATCH --output=/home/users/riah/project_thesis/logs/step3b_scoring_extract_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/step3b_scoring_extract_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false

cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring

echo "--- STEP 3b: Extract Candidate Chunks from TopN (SCORING PIPELINE) ---"
echo "Input: hybrid_train/test_data_topn_scored_5_with_scoring.json"
echo "Output: hybrid_all_candiate_chunks_5_from_topn_with_scoring.txt"
python3 -u extract_candidates_from_topn_with_scoring.py
echo "✅ Step 3b Completed."
