#!/bin/bash
#
# Submit apply_chunk_scorer (scored + scored_ner), S-HRT, and trainer jobs with dependencies.
# - apply scored (dds8) -> s_hrt scored (afterok) -> train scored S-HRT (afterok)
# - apply scored_ner (dds9) -> s_hrt scored_ner (afterok) -> train scored_ner S-HRT (afterok)
#
# Trainer inputs: S-HRT 1-example (*_s_hrt.json) from 8_make_s_hrt_for_chunk_scorer_improved.
#
# Usage: ./submit_apply_and_s_hrt_with_deps.sh
# Or:    bash slurm/submit_apply_and_s_hrt_with_deps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIOMEDRAG="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER_SLURM="$BIOMEDRAG/3_trainning_triple_model"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting apply_chunk_scorer + S-HRT + trainer (with dependencies)"
echo "=========================================="
echo ""

# 1. Apply chunk scorer - scored (dds8)
echo "Submitting apply_chunk_scorer_scored..."
J_APPLY_SCORED=$(sbatch apply_chunk_scorer_scored.slurm | awk '{print $4}')
echo "  -> Job ID: $J_APPLY_SCORED"
echo ""

# 2. Apply chunk scorer - scored_ner (dds9)
echo "Submitting apply_chunk_scorer_scored_ner..."
J_APPLY_SCORED_NER=$(sbatch apply_chunk_scorer_scored_ner.slurm | awk '{print $4}')
echo "  -> Job ID: $J_APPLY_SCORED_NER"
echo ""

# 3. S-HRT scored (depends on apply scored)
echo "Submitting s_hrt_scored (dependency: afterok:$J_APPLY_SCORED)..."
J_SHRT_SCORED=$(sbatch --dependency=afterok:$J_APPLY_SCORED run_s_hrt_scored.slurm | awk '{print $4}')
echo "  -> Job ID: $J_SHRT_SCORED"
echo ""

# 4. S-HRT scored_ner (depends on apply scored_ner)
echo "Submitting s_hrt_scored_ner (dependency: afterok:$J_APPLY_SCORED_NER)..."
J_SHRT_SCORED_NER=$(sbatch --dependency=afterok:$J_APPLY_SCORED_NER run_s_hrt_scored_ner.slurm | awk '{print $4}')
echo "  -> Job ID: $J_SHRT_SCORED_NER"
echo ""

# 5. Trainer scored S-HRT (1-example) on dds8 (depends on s_hrt scored)
echo "Submitting train_scored_s_hrt_1024 (dependency: afterok:$J_SHRT_SCORED)..."
J_TRAIN_SCORED=$(sbatch --dependency=afterok:$J_SHRT_SCORED "$TRAINER_SLURM/slurm_train_scored_s_hrt_1024_dds8.slurm" | awk '{print $4}')
echo "  -> Job ID: $J_TRAIN_SCORED"
echo ""

# 6. Trainer scored_ner S-HRT (1-example) on dds9 (depends on s_hrt scored_ner)
echo "Submitting train_scored_ner_s_hrt_1024 (dependency: afterok:$J_SHRT_SCORED_NER)..."
J_TRAIN_SCORED_NER=$(sbatch --dependency=afterok:$J_SHRT_SCORED_NER "$TRAINER_SLURM/slurm_train_scored_ner_s_hrt_1024_dds9.slurm" | awk '{print $4}')
echo "  -> Job ID: $J_TRAIN_SCORED_NER"
echo ""

echo "=========================================="
echo "All jobs submitted"
echo "=========================================="
echo "  apply_chunk_scorer_scored:     $J_APPLY_SCORED (dds8)"
echo "  apply_chunk_scorer_scored_ner: $J_APPLY_SCORED_NER (dds9)"
echo "  s_hrt_scored:                  $J_SHRT_SCORED (after $J_APPLY_SCORED)"
echo "  s_hrt_scored_ner:              $J_SHRT_SCORED_NER (after $J_APPLY_SCORED_NER)"
echo "  train_scored_s_hrt_1024:       $J_TRAIN_SCORED (after $J_SHRT_SCORED, dds8)"
echo "  train_scored_ner_s_hrt_1024:   $J_TRAIN_SCORED_NER (after $J_SHRT_SCORED_NER, dds9)"
echo ""
echo "Trainer inputs: S-HRT 1-example (*_s_hrt.json)"
echo "Monitor: squeue -u \$USER"
echo "Logs:    .../logs/apply_chunk_scorer_*.out, s_hrt_*.out, train_*_s_hrt_*.out"
echo "=========================================="
