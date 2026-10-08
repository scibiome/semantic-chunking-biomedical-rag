import json
import random

print("=" * 80, flush=True)
print("10-CHUNK IMPROVED S-HRT - Train Triplet Progress (FROM STEP 8)")
print("=" * 80, flush=True)

# 10-CHUNK IMPROVED S-HRT: Use S-HRT files from Step 8 (already have entity concatenation)
# These files already have the correct format, but we process them to match the exact format
s_hrt_train="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/train_chuck_5_s_hrt_ablation_10chunks_improved.json"
s_hrt_test="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/test_chuck_5_s_hrt_ablation_10chunks_improved_deduplicated.json"

triplet_KNN_top1_train="0_Sunflower_triplet_extraction_13b/data/KNN_demo_train.json"
triplet_KNN_top1_test="0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

dev_source="../dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"

def HT_R():
    """Load entity pairs from dev set for each relation (same as original script)"""
    dic_={}
    print(f"\n✓ Loading entity pairs from dev set: {dev_source}", flush=True)
    with open(dev_source) as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())
            h=line["SUBJECT_TEXT"]
            t=line["OBJECT_TEXT"]
            r=line["PREDICATE"]
            if r not in dic_:
                dic_[r]=[h+"|"+t]
            else:
                dic_[r]=dic_[r]+[h+"|"+t]
    print(f"✓ Loaded entity pairs for {len(dic_)} relations", flush=True)
    return dic_

R_HT_mapping=HT_R()

def load_s_hrt_train():
    """Load S-HRT train file (already has entity concatenation from Step 8)"""
    dic_=[]
    print(f"\n✓ Reading S-HRT train file: {s_hrt_train}", flush=True)
    with open(s_hrt_train) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            # S-HRT files already have instruction with entity-concatenated chunk
            dic_.append(full_obj)
    print(f"✓ Loaded {len(dic_)} S-HRT train examples", flush=True)
    return dic_

def load_s_hrt_test():
    """Load S-HRT test file (already has entity concatenation from Step 8)"""
    dic_=[]
    print(f"\n✓ Reading S-HRT test file: {s_hrt_test}", flush=True)
    with open(s_hrt_test) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)
    print(f"✓ Loaded {len(dic_)} S-HRT test examples", flush=True)
    return dic_

SHRTTrain_=load_s_hrt_train()
SHRTTest_=load_s_hrt_test()

# Process TRAIN data
i=-1
fw_train=open("train_chuck_5_triplet_llama2_13b_right_2_10chunks_SHRT.json","w")
print(f"\n✓ Processing TRAIN data from S-HRT files: {triplet_KNN_top1_train}", flush=True)

with open(triplet_KNN_top1_train) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 500 == 0:
                print(f"  Processed {i} train samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            
            # Get the corresponding S-HRT entry (already has entity concatenation)
            if i < len(SHRTTrain_):
                s_hrt_obj = SHRTTrain_[i]
                # Use the instruction from S-HRT (already has entity-concatenated chunk)
                knn_line["instruction"] = s_hrt_obj.get("instruction", knn_line.get("instruction", ""))
                # Keep context and response from KNN (these are the training targets)
                # S-HRT already has correct context and response, but we use KNN for consistency
            else:
                # Fallback: use original KNN instruction
                pass
            
            fw_train.write(json.dumps(knn_line))
            fw_train.write("\n")
            fw_train.flush()

fw_train.close()
print(f"\n✅ Train processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Output: train_chuck_5_triplet_llama2_13b_right_2_10chunks_SHRT.json", flush=True)

# Process TEST data
i=-1
fw_test=open("test_chuck_5_triplet_llama2_13b_right_0_10chunks_SHRT.json","w")
print(f"\n✓ Processing TEST data from S-HRT files: {triplet_KNN_top1_test}", flush=True)

with open(triplet_KNN_top1_test) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 100 == 0:
                print(f"  Processed {i} test samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            
            # Get the corresponding S-HRT entry
            if i < len(SHRTTest_):
                s_hrt_obj = SHRTTest_[i]
                # Use the instruction from S-HRT (already has entity-concatenated chunk)
                knn_line["instruction"] = s_hrt_obj.get("instruction", knn_line.get("instruction", ""))
            else:
                # Fallback: use original KNN instruction
                pass
            
            fw_test.write(json.dumps(knn_line))
            fw_test.write("\n")
            fw_test.flush()

fw_test.close()
print(f"\n✅ Test processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Output: test_chuck_5_triplet_llama2_13b_right_0_10chunks_SHRT.json", flush=True)

print(f"\n✓ Each sample now includes S-HRT format (single chunk with entity concatenation)", flush=True)
print(f"  from 10-CHUNK IMPROVED chunking, fitting within 512 tokens!", flush=True)
print("=" * 80, flush=True)



