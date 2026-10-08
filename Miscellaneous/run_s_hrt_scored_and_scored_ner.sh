#!/bin/bash
# Run S-HRT augmentation for scored and scored_ner apply_chunk_scorer outputs.
# Inputs: train/test_triplet_chunk_scorer_scored.json, train/test_triplet_chunk_scorer_scored_ner.json
# Outputs: *_scored_s_hrt.json, *_scored_s_hrt3.json, *_scored_ner_s_hrt.json, *_scored_ner_s_hrt3.json

set -euo pipefail

cd "$(dirname -- "$0")"
SCRIPT_DIR="$(pwd)"

SCORED_TRAIN="train_triplet_chunk_scorer_scored.json"
SCORED_TEST="test_triplet_chunk_scorer_scored.json"
SCORED_NER_TRAIN="train_triplet_chunk_scorer_scored_ner.json"
SCORED_NER_TEST="test_triplet_chunk_scorer_scored_ner.json"

echo "=========================================="
echo "S-HRT for Scored & Scored NER"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Check inputs exist and are non-empty (apply_chunk_scorer jobs must have completed)
echo "Checking apply_chunk_scorer outputs..."
for f in "$SCORED_TRAIN" "$SCORED_TEST" "$SCORED_NER_TRAIN" "$SCORED_NER_TEST"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f (run apply_chunk_scorer scored/scored_ner jobs first)"
        exit 1
    fi
    if [ ! -s "$f" ]; then
        echo "ERROR: $f is empty (apply_chunk_scorer may not have finished)"
        exit 1
    fi
done
echo "  All inputs found and non-empty"
wc -l "$SCORED_TRAIN" "$SCORED_TEST" "$SCORED_NER_TRAIN" "$SCORED_NER_TEST"
echo ""

# Scored
echo "----------------------------------------"
echo "S-HRT: scored"
echo "----------------------------------------"
python3 8_make_s_hrt_for_chunk_scorer_improved.py \
    --train-input "$SCORED_TRAIN" \
    --test-input "$SCORED_TEST" \
    --dev-source "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"
echo ""

# Scored NER
echo "----------------------------------------"
echo "S-HRT: scored_ner"
echo "----------------------------------------"
python3 8_make_s_hrt_for_chunk_scorer_improved.py \
    --train-input "$SCORED_NER_TRAIN" \
    --test-input "$SCORED_NER_TEST" \
    --dev-source "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"
echo ""

echo "=========================================="
echo "S-HRT complete"
echo "=========================================="
echo "Outputs:"
for base in "train_triplet_chunk_scorer_scored" "test_triplet_chunk_scorer_scored" \
            "train_triplet_chunk_scorer_scored_ner" "test_triplet_chunk_scorer_scored_ner"; do
    for suf in "_s_hrt.json" "_s_hrt3.json"; do
        f="${base}${suf}"
        if [ -f "$f" ]; then
            echo "  $(ls -lh "$f" | awk '{print $9, $5}')"
        fi
    done
done
echo "End time: $(date)"
