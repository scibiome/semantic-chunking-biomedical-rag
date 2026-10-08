# DDI Fixed Chunking - Training Data Generation

This directory contains scripts to generate the final training data for DDI with **fixed chunking**.

## Files Created

- `chuck_triplet_progress_train_ddi_fixed.py` - Process train data
- `chuck_triplet_progress_test_ddi_fixed.py` - Process test data

## What This Step Does

This step combines:
1. **Instruction files** (from Step 5) - Contains chunks with examples
2. **KNN files** (`KNN_ddi_train/test.json`) - Base training data

To create the **final training data** format by:
- Extracting chunk-based examples
- Augmenting them with entity pairs from dev set
- Combining with KNN base instructions

## Expected Outputs

- `train_chuck_5_triplet_ddi_fixed.json` (~1,027 samples)
- `test_chuck_5_triplet_ddi_fixed.json` (~1,094 samples)

## To Run

```bash
cd /home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/ddi_fixed

# Train
python chuck_triplet_progress_train_ddi_fixed.py

# Test  
python chuck_triplet_progress_test_ddi_fixed.py
```

Time: ~5-10 minutes each

## Relations

DDI has 4 relation types:
- MECHANISM
- EFFECT
- ADVISE
- INT

Note: NONE relations have been filtered out from training data.










