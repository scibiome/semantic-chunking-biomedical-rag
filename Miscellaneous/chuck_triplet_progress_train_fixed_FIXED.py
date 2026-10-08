import json

print("=" * 80, flush=True)
print("FIXED CHUNKING - Train Triplet Progress (FIXED to Preserve Chunks)", flush=True)
print("=" * 80, flush=True)

# FIXED CHUNKING: Use fixed INSTRUCTION files (from step 5)
# Note: Path points to the standard Fixed directory
relation_chuck_data_train="../0_make_relation_chuck_and_scorer_data/train_chuck_instruction_5_llama13b_right.json"

triplet_KNN_top1_train="0_Sunflower_triplet_extraction_13b/data/KNN_demo_train.json"

def Relation_Chuck_train():
    """Load instruction files with chunks from Step 5"""
    dic_=[]
    print(f"\n✓ Reading instruction file with chunks: {relation_chuck_data_train}", flush=True)
    with open(relation_chuck_data_train) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)  # Keep full object
    print(f"✓ Loaded {len(dic_)} instruction examples with chunks", flush=True)
    return dic_

RCTrain_=Relation_Chuck_train()

i=-1

# FIXED: Output to fixed-specific file
# Note: This filename must match what trainer_fixed.py expects
fw=open("train_chuck_5_triplet_llama2_13b_right_2_FIXED.json","w")
print(f"\n✓ Processing KNN data and adding chunk-based instructions: {triplet_KNN_top1_train}", flush=True)

with open(triplet_KNN_top1_train) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 500 == 0:
                print(f"  Processed {i} samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            
            # Get the corresponding instruction file entry
            instr_obj = RCTrain_[i]
            
            # Use the chunk-rich instruction
            knn_line["instruction"] = instr_obj["instruction"]
            
            fw.write(json.dumps(knn_line))
            fw.write("\n")
            fw.flush()

fw.close()
print(f"\n✅ Train processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Output: train_chuck_5_triplet_llama2_13b_right_2_FIXED.json", flush=True)
print("=" * 80, flush=True)

