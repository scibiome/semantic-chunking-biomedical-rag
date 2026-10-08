#!/bin/bash
#SBATCH --job-name=step2_scoring_test_ner
#SBATCH --output=/home/users/riah/project_thesis/logs/step2_scoring_test_ner_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/step2_scoring_test_ner_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail
source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag
export TOKENIZERS_PARALLELISM=false

cd /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring

echo "--- STEP 2: Make Test Chunks (SCORING PIPELINE + NER) ---"
echo "Input: hybrid_gm-ciht_relation_with_chunks_5_with_scoring_ner.json"
echo "Embeddings: hybrid_stored_chucks_with_relation_5_with_scoring_ner.txt / hybrid_retrived_chucks_5_with_scoring_ner.npy"
echo "Output: hybrid_test_data_chucks_5_with_scoring_ner.json"
python3 -u hybrid_2_make_test_chunks_with_scoring_ner.py
echo "✅ Step 2 Test (NER) Completed."
