import json
import random


relation_chuck_data_train="train_chuck_instruction_5_llama13b_right_chemprot.json"
relation_chuck_data_test="test_chuck_instruction_5_llama13b_right_chemprot.json"

triplet_KNN_top1_train="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_chemprot_train.json"
triplet_KNN_top1_test="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_chemprot_test.json"

dev_source="/home/users/riah/project_thesis/BIoMedRAG/dataset/2_chemprot/dev_triples_gmciht_format.jsonl"

def HT_R():
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
    return dic_

R_HT_mapping=HT_R()


def Relation_Chuck_train():
    dic_=[]
    with open(relation_chuck_data_train) as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())["instruction"].split("Examples:")[1]
            dic_.append(line)
            # print(line)
            # break
            
    return dic_

RCTrain_=Relation_Chuck_train()

def Relation_Chuck_test():
    dic_=[]
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            line=json.loads(line.strip())["instruction"].split("Examples:")[1]
            dic_.append(line)
            # print(line)
            # break
            
    return dic_

RCTest_=Relation_Chuck_test()



# Generate S-HRT for TEST
print("=" * 70)
print("Step 8: Creating final training data (S-HRT) for ChemProt")
print("=" * 70)
print("\n📊 Processing TEST data...")
i=-1
j=-1
fw=open("test_chuck_5_s_hrt_chemprot.json","w")
with open(triplet_KNN_top1_test) as fr:
    for line in fr.readlines():
        i=i+1
        line=json.loads(line.strip())
        line_=line["instruction"].split("Examples:")
        excamples=line_[1].replace(" 1. Context:Tomudex treatment resulted in increase in cyclin E expression. Response: Tomudex|ACTIVATOR|cyclin E. 2. Context:Dopamine acts as an agonist at D2 receptors. Response: Dopamine|AGONIST|D2 receptors","")
        instruction_=line_[0].strip()
        ex=RCTest_[i]
        sentence_=ex.split("Response")[0]
        renew_exc=[]
        if sentence_ not in excamples:
            j=j+1
            ex=ex.strip().split("Context:")
            for excample in ex[1:2]:
                if len(excample)!=0:
                    excample=excample.split("Response:")
                    input_=excample[0]
                    response_=excample[1].replace(" 2. ","").strip()
                    # Extract relation from H|R|T; fallback to whole string trimmed
                    parts=[p.strip().strip('.') for p in response_.split("|")]
                    rel=parts[1] if len(parts)>=2 else response_.strip().strip('.')
                    hts=R_HT_mapping.get(rel)
                    if not hts:
                        continue
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
            line["instruction"]=line["instruction"].replace(" 1. Context:Tomudex treatment resulted in increase in cyclin E expression. Response: Tomudex|ACTIVATOR|cyclin E. 2. Context:Dopamine acts as an agonist at D2 receptors. Response: Dopamine|AGONIST|D2 receptors","").replace("   "," ")
            fw.write(json.dumps(line))
            fw.write("\n")
            fw.flush()
fw.close()
print(f"✅ Test S-HRT created: {j+1} examples with chunks")
print(f"   Output: test_chuck_5_s_hrt_chemprot.json")

# Generate S-HRT for TRAIN
print("\n📊 Processing TRAIN data...")
i=-1
j=-1
fw_train=open("train_chuck_5_s_hrt_chemprot.json","w")
with open(triplet_KNN_top1_train) as fr:
    for line in fr.readlines():
        i=i+1
        line=json.loads(line.strip())
        line_=line["instruction"].split("Examples:")
        excamples=line_[1].replace(" 1. Context:Tomudex treatment resulted in increase in cyclin E expression. Response: Tomudex|ACTIVATOR|cyclin E. 2. Context:Dopamine acts as an agonist at D2 receptors. Response: Dopamine|AGONIST|D2 receptors","")
        instruction_=line_[0].strip()
        ex=RCTrain_[i]
        sentence_=ex.split("Response")[0]
        renew_exc=[]
        if sentence_ not in excamples:
            j=j+1
            ex=ex.strip().split("Context:")
            for excample in ex[1:2]:
                if len(excample)!=0:
                    excample=excample.split("Response:")
                    input_=excample[0]
                    response_=excample[1].replace(" 2. ","").strip()
                    # Extract relation from H|R|T; fallback to whole string trimmed
                    parts=[p.strip().strip('.') for p in response_.split("|")]
                    rel=parts[1] if len(parts)>=2 else response_.strip().strip('.')
                    hts=R_HT_mapping.get(rel)
                    if not hts:
                        continue
                    ht=random.sample(hts,1)[0].split("|")
                    renew_input=ht[0] +""+input_+""+ ht[1]
                    renew_response=ht[0] +"|"+response_+"|"+ ht[1]
                    new_e="Context: "+renew_input+" Response: "+renew_response
                    renew_exc.append(new_e)
            renew_exc=" ".join(renew_exc)
            new_instruction=instruction_+" "+renew_exc
            line["instruction"]=new_instruction
            fw_train.write(json.dumps(line))
            fw_train.write("\n")
            fw_train.flush()
        else:
            line["instruction"]=line["instruction"].replace(" 1. Context:Tomudex treatment resulted in increase in cyclin E expression. Response: Tomudex|ACTIVATOR|cyclin E. 2. Context:Dopamine acts as an agonist at D2 receptors. Response: Dopamine|AGONIST|D2 receptors","").replace("   "," ")
            fw_train.write(json.dumps(line))
            fw_train.write("\n")
            fw_train.flush()
fw_train.close()
print(f"✅ Train S-HRT created: {j+1} examples with chunks")
print(f"   Output: train_chuck_5_s_hrt_chemprot.json")

print("\n✅ Step 8 completed successfully!")
print("=" * 70)
