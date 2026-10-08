# Testing Entity-Aware Chunk Scorer

## What This Tests

This test version (`apply_chunk_scorer_entity_aware_TEST.py`) adds **entity-aware scoring** to boost chunks that contain the target entity pairs.

## Changes Made

1. **Added `_compute_entity_bonus()` function**:
   - Checks if chunk contains target head entity → +0.15 bonus
   - Checks if chunk contains target tail entity → +0.15 bonus
   - Exact entity pair match → +0.20 additional bonus
   - Relation match → +0.05 bonus
   - Maximum bonus: 0.5 (to avoid overwhelming base score)

2. **Modified `score_candidate()`**:
   - Now returns: `base_score + entity_bonus`
   - Base score is still computed the same way (cosine similarity)
   - Entity bonus is added on top

3. **Output files**:
   - Train: `train_triplet_chunk_scorer_ddi_entity_aware.json`
   - Test: `test_triplet_chunk_scorer_ddi_entity_aware.json`

## How to Test

### Step 1: Run the Test Version

```bash
cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data

python3 apply_chunk_scorer_entity_aware_TEST.py \
    --checkpoint /home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data/1_train_chuck_scorer_model/Our_model_ddi_fixed/checkpoint-1000 \
    --topn-train ../0_make_relation_chuck_and_scorer_data_DDI_FIXED/train_data_topn_chucks_5_ddi.json \
    --topn-test ../0_make_relation_chuck_and_scorer_data_DDI_FIXED/test_data_topn_chucks_5_ddi.json \
    --knn-train ./KNN_ddi_train.json \
    --knn-test ./KNN_ddi_test.json \
    --max-examples 3 \
    --max-length 1024
```

### Step 2: Compare Results

Compare the selected chunks for the same sentence with different target triplets:

```python
import json

# Original version
with open('test_triplet_chunk_scorer_ddi_fixed_s_hrt.json', 'r') as f:
    original = [json.loads(line) for line in f.readlines()]

# Entity-aware version
with open('test_triplet_chunk_scorer_ddi_entity_aware.json', 'r') as f:
    entity_aware = [json.loads(line) for line in f.readlines()]

# Find examples with same context
target_context = "Other CNS depressant drugs (e.g. barbiturates, tranquilizers, opioids and general anesthetics) have additive or potentiating effects with INAPSINE."

original_examples = [ex for ex in original if ex['context'].startswith("Other CNS depressant drugs")]
entity_aware_examples = [ex for ex in entity_aware if ex['context'].startswith("Other CNS depressant drugs")]

print("Original version:")
for ex in original_examples[:3]:
    print(f"  GT: {ex['response']}")
    chunks = ex.get('retrieved_chunks', [])[:2]
    for chunk in chunks:
        print(f"    - {chunk.get('chunk', '')[:60]}... | Score: {chunk.get('score', 0):.4f}")

print("\nEntity-aware version:")
for ex in entity_aware_examples[:3]:
    print(f"  GT: {ex['response']}")
    chunks = ex.get('retrieved_chunks', [])[:2]
    for chunk in chunks:
        print(f"    - {chunk.get('chunk', '')[:60]}... | Score: {chunk.get('score', 0):.4f}")
```

### Step 3: Check if Chunks are More Entity-Specific

**Expected improvement**:
- For `barbiturates|EFFECT|INAPSINE`: Chunks should contain "barbiturates" or "INAPSINE"
- For `tranquilizers|EFFECT|INAPSINE`: Chunks should contain "tranquilizers" or "INAPSINE"
- Chunks should be **different** for different entity pairs

**If it works**:
- Different entity pairs → Different top chunks
- Chunks contain target entities
- Instructions are more entity-specific

## What to Look For

1. **Chunk Selection**:
   - ✅ Chunks should be different for different entity pairs
   - ✅ Chunks should contain target entities

2. **Score Differences**:
   - ✅ Chunks with matching entities should have higher scores
   - ✅ Score differences should be noticeable (0.1-0.3 range)

3. **Instruction Examples**:
   - ✅ Examples should be more entity-specific
   - ✅ Examples should guide toward correct entity pair

## Next Steps

If the test shows improvement:
1. Use entity-aware version for training/generation
2. Compare F1 scores with original version
3. If better, consider making it the default

If the test doesn't show improvement:
1. Adjust bonus weights (currently 0.15 per entity)
2. Try different matching strategies
3. Consider alternative approaches



