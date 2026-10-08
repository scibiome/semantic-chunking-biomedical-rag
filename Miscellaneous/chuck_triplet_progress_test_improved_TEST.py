import json

print("=" * 80, flush=True)
print("IMPROVED CHUNKING - Test Triplet Progress (TEST - Preserves Chunks)", flush=True)
print("=" * 80, flush=True)

# IMPROVED CHUNKING: Use improved INSTRUCTION files (from step 5)
relation_chuck_data_test="../0_make_relation_chuck_and_scorer_data_IMPROVED/test_chuck_instruction_5_llama13b_right_improved_deduplicated.json"

triplet_KNN_top1_test="0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

def Relation_Chuck_test():
    """Load instruction files with chunks from Step 5"""
    dic_=[]
    print(f"\n✓ Reading instruction file with chunks: {relation_chuck_data_test}", flush=True)
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)  # Keep full object, not just examples
    print(f"✓ Loaded {len(dic_)} instruction examples with chunks", flush=True)
    return dic_

RCTest_=Relation_Chuck_test()

i=-1

# IMPROVED TEST: Output to test file (separate from production)
fw=open("test_chuck_5_triplet_llama2_13b_right_0_improved_TEST.json","w")
print(f"\n✓ Processing KNN data and adding chunk-based instructions: {triplet_KNN_top1_test}", flush=True)

with open(triplet_KNN_top1_test) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 500 == 0:
                print(f"  Processed {i} samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            
            # Get the corresponding instruction file entry (which has chunks!)
            instr_obj = RCTest_[i]
            
            # Use the chunk-rich instruction from Step 5 output
            # This preserves all chunks from IMPROVED semantic chunking
            knn_line["instruction"] = instr_obj["instruction"]
            
            # Keep other fields from KNN (context, response, category)
            # These are the actual test targets
            
            fw.write(json.dumps(knn_line))
            fw.write("\n")
            fw.flush()

fw.close()
print(f"\n✅ Test processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Output: test_chuck_5_triplet_llama2_13b_right_0_improved_TEST.json", flush=True)
print(f"   ✓ Each sample now includes chunk-based instruction examples from IMPROVED chunking!", flush=True)
print("=" * 80, flush=True)








