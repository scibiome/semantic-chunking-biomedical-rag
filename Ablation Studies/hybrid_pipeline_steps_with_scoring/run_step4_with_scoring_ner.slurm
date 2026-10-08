#!/bin/bash
#SBATCH --job-name=step4_scoring_emb_ner
#SBATCH --output=/home/users/riah/project_thesis/logs/step4_scoring_emb_ner_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/step4_scoring_emb_ner_%j.err
#SBATCH --time=12:00:00
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

echo "--- STEP 4: Generate Candidate Embeddings (SCORING PIPELINE - NER) ---"
echo "Input: hybrid_train/test_data_topn_scored_5_with_scoring_ner.json"
echo "Output: hybrid_all_candiate_chunks_5_with_scoring_ner.txt"
echo "Shards: shards_scoring_ner/hybrid_all_posible_chuck_vector_5_with_scoring_ner_shard_*.npy"
python3 -u hybrid_4_make_candidate_embeddings_with_scoring_ner.py
echo "✅ Step 4 Completed (NER)."
