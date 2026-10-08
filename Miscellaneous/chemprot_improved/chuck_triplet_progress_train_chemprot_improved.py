import json
import random

print("=" * 80, flush=True)
print("ChemProt IMPROVED SEMANTIC - Train Triplet Progress", flush=True)
print("=" * 80, flush=True)

# IMPROVED SEMANTIC: Use INSTRUCTION files from Step 5
relation_chuck_data_train="../../0_make_relation_chuck_and_scorer_data_CHEMPROT_IMPROVED/train_chuck_instruction_5_llama13b_right_chemprot.json"

triplet_KNN_top1_train="../KNN_chemprot_train.json"

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
    for rel, hts in dic_.items():
        print(f"  {rel}: {len(hts)} examples", flush=True)
    return dic_

R_HT_mapping=HT_R()


def Relation_Chuck_train():
    """Extract chunk examples from instruction files"""
    dic_=[]
    print(f"\n✓ Reading instruction file: {relation_chuck_data_train}", flush=True)
    with open(relation_chuck_data_train) as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())["instruction"].split("Examples:")[1]
            dic_.append(line)
    print(f"✓ Loaded {len(dic_)} instruction examples", flush=True)
    return dic_

RCTrain_=Relation_Chuck_train()


i=-1
j=-1

# IMPROVED: Output to ChemProt improved-specific file
fw=open("train_chuck_5_triplet_chemprot_improved.json","w")
print(f"\n✓ Processing KNN data: {triplet_KNN_top1_train}", flush=True)
with open(triplet_KNN_top1_train) as fr:
        for line in fr.readlines():
            i=i+1
            if i % 500 == 0:
                print(f"  Processed {i} samples...", flush=True)
            
            line=json.loads(line.strip())
            line_=line["instruction"].split("Examples:")
            excamples=line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","")
            instruction_=line_[0].strip()
            ex=RCTrain_[i]

            sentence_=ex.split("Response")[0]
            renew_exc=[]
            if sentence_ not in excamples:
                j=j+1
                ex=ex.strip().split("Context:")

                for excample in ex:
                    if len(excample)!=0 and "Response:" in excample:
                        excample=excample.split("Response:")
                        input_=excample[0]
                        response_=excample[1].strip()
                        # Extract relation from full triple (entity|RELATION|entity)
                        response_parts = response_.split("|")
                        if len(response_parts) >= 2:
                            response_ = response_parts[1].strip()  # Get the RELATION part
                        else:
                            # Skip if not proper triple format or empty
                            continue
                        # Skip empty or placeholder responses
                        if not response_ or response_ in ['{}', '{}.']:
                            continue
                        
                        # Check if relation exists in mapping
                        if response_ not in R_HT_mapping:
                            print(f"Warning: Relation '{response_}' not in mapping, skipping", flush=True)
                            continue
                        
                        hts=R_HT_mapping[response_]
                        ht=random.sample(hts,1)[0].split("|")

                        renew_input=ht[0] +""+input_+""+ ht[1]
                        renew_response=ht[0] +"|"+response_+"|"+ ht[1]


                        new_e="Context: "+renew_input+" Response: "+renew_response
                        renew_exc.append(new_e)
                renew_exc=" ".join(renew_exc)
                new_instruction=instruction_+" "+renew_exc

                line["instruction"]=new_instruction
                fw.write(json.dumps(line))
                fw.write("\n")
                fw.flush()
            else:
                line["instruction"]=line["instruction"].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
                fw.write(json.dumps(line))
                fw.write("\n")
                fw.flush()
fw.close()
print(f"\n✅ Train processing completed!", flush=True)
print(f"   Total samples: {i+1}", flush=True)
print(f"   Modified examples: {j}", flush=True)
print(f"   Output: train_chuck_5_triplet_chemprot_improved.json", flush=True)

