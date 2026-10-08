import json

print("=" * 80, flush=True)
print("10-CHUNK IMPROVED - Train Triplet Progress (Preserve Chunks)", flush=True)
print("=" * 80, flush=True)

# 10-CHUNK IMPROVED: Use 10-chunk instruction files (from step 5)
relation_chuck_data_train="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/train_chuck_instruction_10_llama13b_right_improved.json"
relation_chuck_data_test="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/test_chuck_instruction_10_llama13b_right_improved.json"

triplet_KNN_top1_train="0_Sunflower_triplet_extraction_13b/data/KNN_demo_train.json"
triplet_KNN_top1_test="0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

def Relation_Chuck_train():
    """Load instruction files with chunks from Step 5"""
    dic_=[]
    print(f"\n✓ Reading TRAIN instruction file with chunks: {relation_chuck_data_train}", flush=True)
    with open(relation_chuck_data_train) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)  # Keep full object, not just examples
    print(f"✓ Loaded {len(dic_)} train instruction examples with chunks", flush=True)
    return dic_

def Relation_Chuck_test():
    """Load instruction files with chunks from Step 5"""
    dic_=[]
    print(f"\n✓ Reading TEST instruction file with chunks: {relation_chuck_data_test}", flush=True)
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)  # Keep full object, not just examples
    print(f"✓ Loaded {len(dic_)} test instruction examples with chunks", flush=True)
    return dic_

RCTrain_=Relation_Chuck_train()
RCTest_=Relation_Chuck_test()

# Process TRAIN data
i=-1
fw_train=open("train_chuck_5_triplet_llama2_13b_right_2_10chunks.json","w")
print(f"\n✓ Processing TRAIN KNN data and adding chunk-based instructions: {triplet_KNN_top1_train}", flush=True)

with open(triplet_KNN_top1_train) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 500 == 0:
                print(f"  Processed {i} train samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            
            # Get the corresponding instruction file entry (which has chunks!)
            instr_obj = RCTrain_[i]
            
            # Use the chunk-rich instruction from Step 5 output
            # This preserves all chunks from 10-CHUNK IMPROVED semantic chunking
            knn_line["instruction"] = instr_obj["instruction"]
            
            # Keep other fields from KNN (context, response, category)
            # These are the actual training targets
            
            fw_train.write(json.dumps(knn_line))
            fw_train.write("\n")
            fw_train.flush()

fw_train.close()
print(f"\n✅ Train processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Output: train_chuck_5_triplet_llama2_13b_right_2_10chunks.json", flush=True)

# Process TEST data
i=-1
fw_test=open("test_chuck_5_triplet_llama2_13b_right_0_10chunks.json","w")
print(f"\n✓ Processing TEST KNN data and adding chunk-based instructions: {triplet_KNN_top1_test}", flush=True)

with open(triplet_KNN_top1_test) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 100 == 0:
                print(f"  Processed {i} test samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            
            # Get the corresponding instruction file entry (which has chunks!)
            instr_obj = RCTest_[i]
            
            # Use the chunk-rich instruction from Step 5 output
            knn_line["instruction"] = instr_obj["instruction"]
            
            fw_test.write(json.dumps(knn_line))
            fw_test.write("\n")
            fw_test.flush()

fw_test.close()
print(f"\n✅ Test processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Output: test_chuck_5_triplet_llama2_13b_right_0_10chunks.json", flush=True)

print(f"\n✓ Each sample now includes chunk-based instruction examples from 10-CHUNK IMPROVED chunking!", flush=True)
print("=" * 80, flush=True)

