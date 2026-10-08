import json

print("=" * 80, flush=True)
print("10-CHUNK IMPROVED - Test Triplet Progress (SIMPLE LIST Format)")
print("=" * 80, flush=True)

# 10-CHUNK IMPROVED: Use 10-chunk instruction file with SIMPLE LIST format
relation_chuck_data_test="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/test_chuck_instruction_10_llama13b_right_improved_SIMPLE_LIST.json"

triplet_KNN_top1_test="0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

def Relation_Chuck_test():
    """Load instruction files with chunks from Step 5 (simple list format)"""
    dic_=[]
    print(f"\n✓ Reading TEST instruction file with chunks: {relation_chuck_data_test}", flush=True)
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)  # Keep full object
    print(f"✓ Loaded {len(dic_)} test instruction examples with chunks", flush=True)
    return dic_

RCTest_=Relation_Chuck_test()

i=-1

# 10-CHUNK IMPROVED: Output to test file with SIMPLE LIST format
fw=open("test_chuck_5_triplet_llama2_13b_right_0_10chunks_improved_SIMPLE_LIST.json","w")
print(f"\n✓ Processing TEST data and adding chunk-based instructions: {triplet_KNN_top1_test}", flush=True)

with open(triplet_KNN_top1_test) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 100 == 0:
                print(f"  Processed {i} samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            
            # Get the corresponding instruction file entry (which has chunks!)
            instr_obj = RCTest_[i]
            
            # Use the chunk-rich instruction from Step 5 output (simple list format)
            # This preserves all chunks from 10-chunk improved semantic chunking
            knn_line["instruction"] = instr_obj["instruction"]
            
            # Keep other fields from KNN (context, response, category)
            # These are the actual test targets
            
            fw.write(json.dumps(knn_line))
            fw.write("\n")
            fw.flush()

fw.close()
print(f"\n✅ Test processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Output: test_chuck_5_triplet_llama2_13b_right_0_10chunks_improved_SIMPLE_LIST.json", flush=True)
print(f"   ✓ Each sample now includes chunk-based instruction examples (SIMPLE LIST format)!", flush=True)
print("=" * 80, flush=True)



