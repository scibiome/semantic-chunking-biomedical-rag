import json
import random

print("=" * 80, flush=True)
print("10-CHUNK IMPROVED - Train Triplet Progress (WITH Entity Concatenation)")
print("=" * 80, flush=True)

# 10-CHUNK IMPROVED: Use 10-chunk instruction files (from step 5)
relation_chuck_data_train="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/train_chuck_instruction_10_llama13b_right_improved.json"
relation_chuck_data_test="../0_make_relation_chuck_and_scorer_data_IMPROVED_10CHUNKS/test_chuck_instruction_10_llama13b_right_improved.json"

triplet_KNN_top1_train="0_Sunflower_triplet_extraction_13b/data/KNN_demo_train.json"
triplet_KNN_top1_test="0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

dev_source="../dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"

def HT_R():
    """Load entity pairs from dev set for each relation (same as fixed script)"""
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
    """Load instruction files with chunks from Step 5"""
    dic_=[]
    print(f"\n✓ Reading TRAIN instruction file with chunks: {relation_chuck_data_train}", flush=True)
    with open(relation_chuck_data_train) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            # Extract Examples section (same as fixed script)
            instruction = full_obj.get("instruction", "")
            if "Examples:" in instruction:
                examples_section = instruction.split("Examples:")[1]
                dic_.append(examples_section)
            else:
                dic_.append("")
    print(f"✓ Loaded {len(dic_)} train instruction examples with chunks", flush=True)
    return dic_

def Relation_Chuck_test():
    """Load instruction files with chunks from Step 5"""
    dic_=[]
    print(f"\n✓ Reading TEST instruction file with chunks: {relation_chuck_data_test}", flush=True)
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            full_obj = json.loads(line.strip())
            instruction = full_obj.get("instruction", "")
            if "Examples:" in instruction:
                examples_section = instruction.split("Examples:")[1]
                dic_.append(examples_section)
            else:
                dic_.append("")
    print(f"✓ Loaded {len(dic_)} test instruction examples with chunks", flush=True)
    return dic_

RCTrain_=Relation_Chuck_train()
RCTest_=Relation_Chuck_test()

# Process TRAIN data (with entity concatenation)
i=-1
j=-1
fw_train=open("train_chuck_5_triplet_llama2_13b_right_2_10chunks_FIXED.json","w")
print(f"\n✓ Processing TRAIN KNN data and adding chunk-based instructions with entity concatenation: {triplet_KNN_top1_train}", flush=True)

with open(triplet_KNN_top1_train) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 500 == 0:
                print(f"  Processed {i} train samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            line_=knn_line["instruction"].split("Examples:")
            excamples=line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","") if len(line_) > 1 else ""
            instruction_=line_[0].strip()
            ex=RCTrain_[i]

            # Follow original script logic exactly (same as chuck_triplet_progress_train.py)
            sentence_=ex.split("Response")[0] if ex else ""
            renew_exc=[]
            if sentence_ and sentence_ not in excamples:
                j=j+1
                # ORIGINAL: Simple split by "Context:" (exactly like original script line 65)
                ex=ex.strip().split("Context:")

                for excample in ex:
                    if len(excample)!=0 and "Response:" in excample:
                        excample=excample.split("Response:")
                        input_=excample[0]
                        response_=excample[1].strip()
                        # Extract relation if in triple format (entity|RELATION|entity)
                        response_parts = response_.split("|")
                        if len(response_parts) >= 2:
                            response_ = response_parts[1].strip()  # Get RELATION from entity|RELATION|entity
                        else:
                            # If no |, assume it's just the relation name
                            response_ = response_.strip()
                        # Skip empty or placeholder responses (like fixed script lines 72-74)
                        if not response_ or response_ in ['{}', '{}.', ''] or response_.startswith('{}'):
                            continue
                        # ORIGINAL: Direct lookup - will crash if not in mapping (same as original line 74)
                        hts=R_HT_mapping[response_]
                        ht=random.sample(hts,1)[0].split("|")

                        # CONCATENATE entities with chunk text (same as fixed script)
                        renew_input=ht[0] +""+input_+""+ ht[1]
                        renew_response=ht[0] +"|"+response_+"|"+ ht[1]

                        new_e="Context: "+renew_input+" Response: "+renew_response
                        renew_exc.append(new_e)
                
                if renew_exc:
                    renew_exc=" ".join(renew_exc)
                    new_instruction=instruction_+" "+renew_exc
                    knn_line["instruction"]=new_instruction
                else:
                    # If no chunks processed, keep original instruction
                    knn_line["instruction"]=instruction_
            else:
                # If chunks already in examples, just clean up
                knn_line["instruction"]=knn_line["instruction"].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
            
            fw_train.write(json.dumps(knn_line))
            fw_train.write("\n")
            fw_train.flush()

fw_train.close()
print(f"\n✅ Train processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Modified examples: {j+1}", flush=True)
print(f"   Output: train_chuck_5_triplet_llama2_13b_right_2_10chunks_FIXED.json", flush=True)

# Process TEST data (with entity concatenation)
i=-1
j=-1
fw_test=open("test_chuck_5_triplet_llama2_13b_right_0_10chunks_FIXED.json","w")
print(f"\n✓ Processing TEST KNN data and adding chunk-based instructions with entity concatenation: {triplet_KNN_top1_test}", flush=True)

with open(triplet_KNN_top1_test) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 100 == 0:
                print(f"  Processed {i} test samples...", flush=True)
            
            knn_line=json.loads(line.strip())
            line_=knn_line["instruction"].split("Examples:")
            excamples=line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","") if len(line_) > 1 else ""
            instruction_=line_[0].strip()
            ex=RCTest_[i]

            # Follow original script logic exactly
            sentence_=ex.split("Response")[0] if ex else ""
            renew_exc=[]
            if sentence_ and sentence_ not in excamples:
                j=j+1
                # ORIGINAL: Simple split by "Context:" (exactly like original script line 65)
                ex=ex.strip().split("Context:")

                for excample in ex:
                    if len(excample)!=0 and "Response:" in excample:
                        excample=excample.split("Response:")
                        input_=excample[0]
                        response_=excample[1].strip()
                        # Extract relation if in triple format (entity|RELATION|entity)
                        response_parts = response_.split("|")
                        if len(response_parts) >= 2:
                            response_ = response_parts[1].strip()  # Get RELATION from entity|RELATION|entity
                        else:
                            # If no |, assume it's just the relation name
                            response_ = response_.strip()
                        # Skip empty or placeholder responses (like fixed script lines 72-74)
                        if not response_ or response_ in ['{}', '{}.', ''] or response_.startswith('{}'):
                            continue
                        # ORIGINAL: Direct lookup - will crash if not in mapping (same as original line 74)
                        hts=R_HT_mapping[response_]
                        ht=random.sample(hts,1)[0].split("|")

                        renew_input=ht[0] +""+input_+""+ ht[1]
                        renew_response=ht[0] +"|"+response_+"|"+ ht[1]

                        new_e="Context: "+renew_input+" Response: "+renew_response
                        renew_exc.append(new_e)
                
                if renew_exc:
                    renew_exc=" ".join(renew_exc)
                    new_instruction=instruction_+" "+renew_exc
                    knn_line["instruction"]=new_instruction
                else:
                    knn_line["instruction"]=instruction_
            else:
                knn_line["instruction"]=knn_line["instruction"].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
            
            fw_test.write(json.dumps(knn_line))
            fw_test.write("\n")
            fw_test.flush()

fw_test.close()
print(f"\n✅ Test processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Modified examples: {j+1}", flush=True)
print(f"   Output: test_chuck_5_triplet_llama2_13b_right_0_10chunks_FIXED.json", flush=True)

print(f"\n✓ Each sample now includes chunk-based instruction examples from 10-CHUNK IMPROVED chunking", flush=True)
print(f"  WITH entity concatenation (same as fixed script) for apple-to-apple comparison!", flush=True)
print("=" * 80, flush=True)

