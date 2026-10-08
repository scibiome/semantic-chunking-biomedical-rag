#!/usr/bin/env python3
"""
Regenerate ChemProt IMPROVED test data WITH chunks
Similar approach to DDI FIXED - directly use Step 5 instruction outputs
"""
import json

print("=" * 80, flush=True)
print("ChemProt IMPROVED - Regenerate TEST with Chunks", flush=True)
print("=" * 80, flush=True)
print()

# Step 5 output WITH chunks
instruction_file = "../../0_make_relation_chuck_and_scorer_data_CHEMPROT_IMPROVED/test_chuck_instruction_5_llama13b_right_chemprot.json"

# KNN file with context/response
knn_file = "../KNN_chemprot_test.json"

# Output file
output_file = "test_chuck_5_triplet_chemprot_improved.json"

print(f"📊 Loading chunk-rich instructions from: {instruction_file}", flush=True)
instructions_with_chunks = []
with open(instruction_file, 'r') as f:
    for line in f:
        obj = json.loads(line.strip())
        instructions_with_chunks.append(obj["instruction"])

print(f"   ✓ Loaded {len(instructions_with_chunks)} instructions with chunks", flush=True)
print()

print(f"📊 Processing KNN data from: {knn_file}", flush=True)
i = 0
with open(knn_file, 'r') as fr, open(output_file, 'w') as fw:
    for line in fr:
        if i % 500 == 0:
            print(f"   Processed {i} samples...", flush=True)
        
        knn_entry = json.loads(line.strip())
        
        # Replace instruction with chunk-rich version from Step 5
        knn_entry["instruction"] = instructions_with_chunks[i]
        
        fw.write(json.dumps(knn_entry))
        fw.write("\n")
        fw.flush()
        
        i += 1

print()
print(f"✅ Test processing completed!", flush=True)
print(f"   Total samples: {i}", flush=True)
print(f"   Output: {output_file}", flush=True)
print()

# Verify chunks
print("🔍 Verifying chunks in output...", flush=True)
with open(output_file, 'r') as f:
    sample = json.loads(f.readline())
    instr = sample['instruction']
    context_count = instr.count('Context:')
    print(f"   Sample instruction length: {len(instr)} chars", flush=True)
    print(f"   Context count: {context_count}", flush=True)
    if context_count > 2:
        print(f"   ✅ HAS CHUNKS: ~{context_count-2} chunk examples", flush=True)
    else:
        print(f"   ❌ NO CHUNKS: Only {context_count} Context occurrences", flush=True)
print()
print("=" * 80, flush=True)









