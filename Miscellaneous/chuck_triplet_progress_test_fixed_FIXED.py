import json

# FIXED CHUNKING: Use fixed INSTRUCTION files (clean)
relation_chuck_data_test = "../0_make_relation_chuck_and_scorer_data/test_chuck_instruction_5_llama13b_right_deduplicated.json"
triplet_KNN_top1_test = "0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

def Relation_Chuck_test():
    dic_ = []
    print(f"\n✓ Reading instruction file with fixed chunks: {relation_chuck_data_test}", flush=True)
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)
    print(f"✓ Loaded {len(dic_)} instruction examples with fixed chunks", flush=True)
    return dic_

RCTest_ = Relation_Chuck_test()

i = -1
fw = open("test_chuck_5_triplet_llama2_13b_right_0_FIXED_deduplicated_FIXED.json", "w")
print(f"\n✓ Processing TEST data and adding clean fixed chunk-based instructions...", flush=True)

with open(triplet_KNN_top1_test) as fr:
    for line in fr.readlines():
        i = i + 1
        knn_line = json.loads(line.strip())
        
        # Get the corresponding instruction file entry (which has chunks!)
        instr_obj = RCTest_[i]
        
        # Direct assignment of clean instruction (preserves chunks, no complex logic)
        knn_line["instruction"] = instr_obj["instruction"]
        
        fw.write(json.dumps(knn_line))
        fw.write("\n")
        
        if i % 500 == 0:
             print(f"  Processed {i} samples...", flush=True)

fw.close()
print(f"\n✅ Generated test_chuck_5_triplet_llama2_13b_right_0_FIXED_deduplicated_FIXED.json with {i+1} examples")



