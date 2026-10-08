#!/usr/bin/env python3
"""Debug the chunk processing logic"""
import json

# Read first sample from each source
print("=" * 80)
print("DEBUG: Chunk Processing Logic")
print("=" * 80)

# ChemProt sources
instr_file = "../0_make_relation_chuck_and_scorer_data_CHEMPROT/train_chuck_instruction_5_llama13b_right_chemprot.json"
knn_file = "KNN_chemprot_train.json"

print("\n1. Original INSTRUCTION file (from Step 5):")
with open(instr_file, 'r') as f:
    instr_obj = json.loads(f.readline())
    instr_full = instr_obj['instruction']
    instr_examples = instr_full.split("Examples:")[1] if "Examples:" in instr_full else ""
    
print(f"   Full instruction length: {len(instr_full)}")
print(f"   Context count: {instr_full.count('Context:')}")
print(f"   Examples part length: {len(instr_examples)}")
print(f"   First 300 chars of examples: {instr_examples[:300]}...")

print("\n2. KNN file (base template):")
with open(knn_file, 'r') as f:
    knn_obj = json.loads(f.readline())
    knn_instr = knn_obj['instruction']
    knn_examples = knn_instr.split("Examples:")[1] if "Examples:" in knn_instr else ""

print(f"   Full instruction length: {len(knn_instr)}")
print(f"   Context count: {knn_instr.count('Context:')}")
print(f"   Examples part length: {len(knn_examples)}")
print(f"   First 300 chars of examples: {knn_examples[:300]}...")

print("\n3. Checking the condition:")
print(f"   instr_examples[:100] = {instr_examples[:100]}")
print(f"   knn_examples[:100] = {knn_examples[:100]}")

sentence_from_instr = instr_examples.split("Response")[0]
print(f"\n   sentence_from_instr = {sentence_from_instr[:100]}...")
print(f"   Is it in knn_examples? {sentence_from_instr in knn_examples}")

print("\n4. THE ISSUE:")
print("   The script checks: if sentence_ NOT in excamples:")
print("   But since instruction file HAS chunks and KNN doesn't,")
print("   the sentence IS in the KNN examples (they're the same base examples),")
print("   so it goes to ELSE branch which REMOVES chunks!")
print("\n   Solution: The logic should be REVERSED or condition changed")

print("=" * 80)
