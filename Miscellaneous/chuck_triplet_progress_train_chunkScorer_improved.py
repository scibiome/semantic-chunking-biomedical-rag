import json
import random

relation_chuck_data_train="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/train_triplet_chunk_scorer_improved_filtered.json"
relation_chuck_data_test="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/test_triplet_chunk_scorer_improved_filtered.json"


triplet_KNN_top1_train="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_train.json"
triplet_KNN_top1_test="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_test.json"

dev_source="/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"

def HT_R():
    dic_={}
    with open(dev_source) as fr:    #"/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"
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


with open(relation_chuck_data_train) as _train_fr:
    SCORER_TRAIN_ENTRIES=[json.loads(line.strip()) for line in _train_fr.readlines()]

def Relation_Chuck_train():
    dic_=[]
    for entry in SCORER_TRAIN_ENTRIES:
        instruction=entry.get("instruction","")
        if "Examples:" in instruction:
            dic_.append(instruction.split("Examples:",1)[1])
        else:
            dic_.append("")
    return dic_

RCTrain_=Relation_Chuck_train()




i=-1
j=-1

fw=open("/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/train_triplet_chunk_scorer_improved_filtered_sorted.json","w")
with open(triplet_KNN_top1_train) as fr:
        for line in fr.readlines():
            i=i+1
            line=json.loads(line.strip())
            scorer_entry=SCORER_TRAIN_ENTRIES[i]
            line_=line["instruction"].split("Examples:")
            excamples=line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","")
            instruction_=line_[0].strip()
            ex=RCTrain_[i]
            # print(excamples)

            sentence_=ex.split("Response")[0]
            renew_exc=[]
            if sentence_ not in excamples:
                j=j+1
                # print(ex)
                ex=ex.strip().split("Context:")
                # print(ex)

                for excample in ex:
                    if len(excample)!=0:
                     
                        excample=excample.split("Response:")
                        if len(excample) < 2:
                            continue  # Skip if no Response part
                        input_=excample[0]
                        response_=excample[1].strip()
                        if response_ not in R_HT_mapping:
                            continue  # Skip if relation not in mapping
                        hts=R_HT_mapping[response_]
                        ht=random.sample(hts,1)[0].split("|")

                        renew_input=ht[0] +""+input_+""+ ht[1]
                        renew_response=ht[0] +"|"+response_+"|"+ ht[1]
                        # print(renew_input)
                        # print(renew_response)
                        new_e="Context: "+renew_input+" Response: "+renew_response
                        renew_exc.append(new_e)
                       
        

                renew_exc=" ".join(renew_exc)
                new_instruction=instruction_+" "+renew_exc
                # print(new_instruction)
                line["instruction"]=new_instruction
                line["retrieved_chunks"]=scorer_entry.get("retrieved_chunks", [])
                fw.write(json.dumps(line))
                fw.write("\n")
                fw.flush()

            else:
                line["instruction"]=line["instruction"].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","").replace("   "," ")
                line["retrieved_chunks"]=scorer_entry.get("retrieved_chunks", [])
                fw.write(json.dumps(line))
                fw.write("\n")
                fw.flush()

print(j)
