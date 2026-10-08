#!/bin/bash
#SBATCH --job-name=s_hrt_scored_ner
#SBATCH --partition=compute
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home/users/riah/project_thesis/logs/s_hrt_scored_scored_ner_%j.out
#SBATCH --error=/home/users/riah/project_thesis/logs/s_hrt_scored_scored_ner_%j.err
#SBATCH --mail-type=NONE

# S-HRT step for scored and scored_ner apply_chunk_scorer outputs.
# Run after apply_chunk_scorer jobs (scored + scored_ner) have completed.
# No GPU required.

echo "=========================================="
echo "S-HRT: Scored & Scored NER"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

source /home/users/riah/miniconda3/etc/profile.d/conda.sh
conda activate biomedrag

cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

bash run_s_hrt_scored_and_scored_ner.sh

EXIT=$?
echo ""
echo "Exit code: $EXIT"
echo "End time: $(date)"
exit $EXIT
