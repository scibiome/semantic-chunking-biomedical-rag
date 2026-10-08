import json

print("=" * 80)
print("IMPROVED CHUNKING - Test Triplet Progress (FIXED to Preserve Chunks)")
print("=" * 80)

# IMPROVED TEST: Use improved INSTRUCTION files (from step 5)
relation_chuck_data_test="../0_make_relation_chuck_and_scorer_data_IMPROVED/test_chuck_instruction_5_llama13b_right_improved_deduplicated.json"

triplet_KNN_top1_test="0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

def Relation_Chuck_test():
    """Load instruction files with chunks from Step 5"""
    dic_=[]
    print(f"\n✓ Reading instruction file with chunks: {relation_chuck_data_test}")
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)  # Keep full object
    print(f"✓ Loaded {len(dic_)} instruction examples with chunks")
    return dic_

RCTest_=Relation_Chuck_test()

i=-1

# IMPROVED: Output to improved-specific file
# Save with a new unique name to avoid overwriting
fw = open("test_chuck_5_triplet_llama2_13b_right_0_improved_deduplicated_FIXED_CLEAN_V3.json", "w")

print(f"\n✓ Processing TEST data and adding clean chunk-based instructions...")

with open(triplet_KNN_top1_test) as fr:
    for line in fr.readlines():
        i=i+1
        if i % 100 == 0:
            print(f"  Processed {i} samples...")
        
        knn_line=json.loads(line.strip())
        
        # Get the corresponding instruction file entry (which has chunks!)
        instr_obj = RCTest_[i]
        
        # Use the chunk-rich instruction from Step 5 output
        knn_line["instruction"] = instr_obj["instruction"]
        
        fw.write(json.dumps(knn_line))
        fw.write("\n")
        fw.flush()

fw.close()
print(f"\n✅ Generated test_chuck_5_triplet_llama2_13b_right_0_improved_deduplicated_FIXED_CLEAN_V3.json with {i+1} examples")
