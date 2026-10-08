import json
import random

print("=" * 80, flush=True)
print("ChemProt FIXED CHUNKING - Test Triplet Progress", flush=True)
print("=" * 80, flush=True)

# FIXED CHUNKING: Use instruction files from Step 5 (contains chunks!)
relation_chuck_data_test="../../0_make_relation_chuck_and_scorer_data_CHEMPROT/test_chuck_instruction_5_llama13b_right_chemprot.json"

triplet_KNN_top1_test="../KNN_chemprot_test.json"

dev_source="../../dataset/2_chemprot/dev_triples_gmciht_format.jsonl"

def HT_R():
    """Build head-tail mapping per relation from dev set"""
    dic_={}
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
    print(f"✓ Loaded {len(dic_)} relations from dev set", flush=True)
    return dic_

R_HT_mapping=HT_R()


def Relation_Chuck_test():
    """Load instruction files with chunks from Step 5"""
    dic_=[]
    print(f"\n✓ Reading test instruction file with chunks: {relation_chuck_data_test}", flush=True)
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            dic_.append(full_obj)  # Keep full object, not just examples
    print(f"✓ Loaded {len(dic_)} instruction examples with chunks", flush=True)
    return dic_

RCTest_=Relation_Chuck_test()


i=-1

# Output to ChemProt fixed-specific file
fw=open("test_chuck_5_triplet_chemprot_fixed.json","w")
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
            # This already has ~50-100 chunk examples per sample
            knn_line["instruction"] = instr_obj["instruction"]
            
            # Keep other fields from KNN (context, response, category)
            
            fw.write(json.dumps(knn_line))
            fw.write("\n")
            fw.flush()

fw.close()
print(f"\n✅ Test processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Output: test_chuck_5_triplet_chemprot_fixed.json", flush=True)
print(f"   ✓ Each sample now includes chunk-based instruction examples!", flush=True)
print("=" * 80, flush=True)
