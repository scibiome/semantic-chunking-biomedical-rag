import json
import random
import re

print("=" * 80, flush=True)
print("10-CHUNK IMPROVED - Direct from Step 5 Instruction Files")
print("=" * 80, flush=True)

# Use Step 5 instruction files directly (skip Step 8)
relation_chuck_data_train="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/train_chuck_instruction_10_llama13b_right_improved.json"
relation_chuck_data_test="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/test_chuck_instruction_10_llama13b_right_improved.json"

triplet_KNN_top1_train="0_Sunflower_triplet_extraction_13b/data/KNN_demo_train.json"
triplet_KNN_top1_test="0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

dev_source="../dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"

def HT_R():
    """Load entity pairs from dev set for each relation"""
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

def Relation_Chuck_train():
    """Load Examples section from Step 5 instruction files"""
    dic_=[]
    print(f"\n✓ Reading Step 5 train instruction file: {relation_chuck_data_train}", flush=True)
    with open(relation_chuck_data_train) as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            instruction = obj.get("instruction", "")
            if "Examples:" in instruction:
                examples = instruction.split("Examples:")[1]
                dic_.append(examples)
            else:
                dic_.append("")
    print(f"✓ Loaded {len(dic_)} train examples from Step 5", flush=True)
    return dic_

RCTrain_=Relation_Chuck_train()

def Relation_Chuck_test():
    """Load Examples section from Step 5 instruction files"""
    dic_=[]
    print(f"\n✓ Reading Step 5 test instruction file: {relation_chuck_data_test}", flush=True)
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            instruction = obj.get("instruction", "")
            if "Examples:" in instruction:
                examples = instruction.split("Examples:")[1]
                dic_.append(examples)
            else:
                dic_.append("")
    print(f"✓ Loaded {len(dic_)} test examples from Step 5", flush=True)
    return dic_

RCTest_=Relation_Chuck_test()

# Process TRAIN data
i=-1
j=-1
fw_train=open("train_chuck_5_triplet_llama2_13b_right_2_10chunks_DIRECT_no_bug.json","w")
print(f"\n✓ Processing TRAIN data: {triplet_KNN_top1_train}", flush=True)

with open(triplet_KNN_top1_train) as fr:
    for line in fr.readlines():
        i=i+1
        if i % 500 == 0:
            print(f"  Processed {i} train samples...", flush=True)
        
        knn_line=json.loads(line.strip())
        line_=knn_line["instruction"].split("Examples:")
        excamples=line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","")
        instruction_=line_[0].strip()
        
        if i < len(RCTrain_):
            ex=RCTrain_[i]
            # Check if Step 5 examples have chunks (Context markers)
            has_chunks = "Context:" in ex and ex.count("Context:") > 2  # More than just template chunks
            
            if has_chunks:
                j=j+1
                # Use split by "Context:" which preserves responses with "|" (like "bicarbonate|INTERACTS_WITH|alcohol")
                # This is better than splitting by "|" which breaks chunks incorrectly
                ex_parts = ex.strip().split("Context:")
                renew_exc=[]
                
                # Process all chunks (skip template chunks at indices 1-2)
                # Index 0 is empty (before first Context:)
                # Index 1 is first template chunk
                # Index 2 is second template chunk  
                # Index 3+ are actual chunks
                for excample in ex_parts[3:]:  # Skip template chunks
                    if len(excample)!=0:
                        excample=excample.split("Response:")
                        if len(excample) >= 2:
                            input_=excample[0].strip()
                            response_=excample[1].strip().replace(" 2. ","").strip()
                            
                            # Extract relation from H|R|T format or just relation
                            parts=[p.strip().strip('.') for p in response_.split("|")]
                            # If response has "|", check if it's H|R|T format or just relation|
                            if len(parts) >= 2:
                                # Could be H|R|T format or just relation| (trailing |)
                                if parts[1]:  # H|R|T format: parts[1] is the relation
                                    rel = parts[1]
                                else:  # Just relation| format: parts[0] is the relation
                                    rel = parts[0] if parts[0] else response_.strip().strip('.')
                            else:
                                # Just relation (no |)
                                rel = response_.strip().strip('.')
                            
                            # Skip if relation not found or empty
                            if not rel or rel not in R_HT_mapping:
                                continue
                            
                            hts=R_HT_mapping[rel]
                            ht=random.sample(hts,1)[0].split("|")
                            
                            renew_input=ht[0] +""+input_+""+ ht[1]
                            renew_response=ht[0] +"|"+rel+"|"+ ht[1]
                            new_e="Context: "+renew_input+" Response: "+renew_response
                            renew_exc.append(new_e)
                
                if renew_exc:  # Only add if we have chunks
                    renew_exc=" ".join(renew_exc)
                    new_instruction=instruction_+" "+renew_exc
                    knn_line["instruction"]=new_instruction
                else:
                    # No valid chunks found, keep base instruction
                    knn_line["instruction"]=instruction_.replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
            else:
                # Step 5 has no chunks, remove template examples
                knn_line["instruction"]=instruction_.replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
        
        fw_train.write(json.dumps(knn_line))
        fw_train.write("\n")
        fw_train.flush()

fw_train.close()
print(f"\n✅ Train processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Samples with chunks processed: {j+1}", flush=True)
print(f"   Output: train_chuck_5_triplet_llama2_13b_right_2_10chunks_DIRECT_no_bug.json", flush=True)

# Process TEST data
i=-1
j=-1
fw_test=open("test_chuck_5_triplet_llama2_13b_right_0_10chunks_DIRECT_no_bug.json","w")
print(f"\n✓ Processing TEST data: {triplet_KNN_top1_test}", flush=True)

with open(triplet_KNN_top1_test) as fr:
    for line in fr.readlines():
        i=i+1
        if i % 100 == 0:
            print(f"  Processed {i} test samples...", flush=True)
        
        knn_line=json.loads(line.strip())
        line_=knn_line["instruction"].split("Examples:")
        excamples=line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","")
        instruction_=line_[0].strip()
        
        if i < len(RCTest_):
            ex=RCTest_[i]
            # Check if Step 5 examples have chunks (Context markers)
            has_chunks = "Context:" in ex and ex.count("Context:") > 2  # More than just template chunks
            
            if has_chunks:
                j=j+1
                # Use split by "Context:" which preserves responses with "|" (like "bicarbonate|INTERACTS_WITH|alcohol")
                # This is better than splitting by "|" which breaks chunks incorrectly
                ex_parts = ex.strip().split("Context:")
                renew_exc=[]
                
                # Process all chunks (skip template chunks at indices 1-2)
                # Index 0 is empty (before first Context:)
                # Index 1 is first template chunk
                # Index 2 is second template chunk  
                # Index 3+ are actual chunks
                for excample in ex_parts[3:]:  # Skip template chunks
                    if len(excample)!=0:
                        excample=excample.split("Response:")
                        if len(excample) >= 2:
                            input_=excample[0].strip()
                            response_=excample[1].strip().replace(" 2. ","").strip()
                            
                            # Extract relation from H|R|T format or just relation
                            parts=[p.strip().strip('.') for p in response_.split("|")]
                            # If response has "|", check if it's H|R|T format or just relation|
                            if len(parts) >= 2:
                                # Could be H|R|T format or just relation| (trailing |)
                                if parts[1]:  # H|R|T format: parts[1] is the relation
                                    rel = parts[1]
                                else:  # Just relation| format: parts[0] is the relation
                                    rel = parts[0] if parts[0] else response_.strip().strip('.')
                            else:
                                # Just relation (no |)
                                rel = response_.strip().strip('.')
                            
                            # Skip if relation not found or empty
                            if not rel or rel not in R_HT_mapping:
                                continue
                            
                            hts=R_HT_mapping[rel]
                            ht=random.sample(hts,1)[0].split("|")
                            
                            renew_input=ht[0] +""+input_+""+ ht[1]
                            renew_response=ht[0] +"|"+rel+"|"+ ht[1]
                            new_e="Context: "+renew_input+" Response: "+renew_response
                            renew_exc.append(new_e)
                
                if renew_exc:  # Only add if we have chunks
                    renew_exc=" ".join(renew_exc)
                    new_instruction=instruction_+" "+renew_exc
                    knn_line["instruction"]=new_instruction
                else:
                    # No valid chunks found, keep base instruction
                    knn_line["instruction"]=instruction_.replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
            else:
                # Step 5 has no chunks, remove template examples
                knn_line["instruction"]=instruction_.replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
        
        fw_test.write(json.dumps(knn_line))
        fw_test.write("\n")
        fw_test.flush()

fw_test.close()
print(f"\n✅ Test processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Samples with chunks processed: {j+1}", flush=True)
print(f"   Output: test_chuck_5_triplet_llama2_13b_right_0_10chunks_DIRECT_no_bug.json", flush=True)

print(f"\n✓ Each sample now includes chunks from Step 5 instruction files", flush=True)
print(f"  with entity concatenation, ready for training!", flush=True)
print("=" * 80, flush=True)



