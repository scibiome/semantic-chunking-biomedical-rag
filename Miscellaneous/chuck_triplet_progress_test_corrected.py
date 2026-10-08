import json
import random

# CORRECTED PATHS - using the expected files that already exist
relation_chuck_data_train = "/home/users/riah/project_thesis/BIoMedRAG/1_train_chuck_scorer_model/data/train_chuck5_s_r_llama2_13b_right_one.json"
relation_chuck_data_test = "/home/users/riah/project_thesis/BIoMedRAG/1_train_chuck_scorer_model/data/test_chuck5_s_r_llama2_13b_right_one.json"

triplet_KNN_top1_train = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_train.json"
triplet_KNN_top1_test = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_test.json"

dev_source = "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"

def HT_R():
    dic_ = {}
    with open(dev_source) as fr:
        for line in fr.readlines():
            line = json.loads(line.strip())
            h = line["SUBJECT_TEXT"]
            t = line["OBJECT_TEXT"]
            r = line["PREDICATE"]
            if r not in dic_:
                dic_[r] = [h + "|" + t]
            else:
                dic_[r] = dic_[r] + [h + "|" + t]
    return dic_

R_HT_mapping = HT_R()

def Relation_Chuck_test():
    dic_ = []
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            line = json.loads(line.strip())["instruction"].split("Examples:")[1]
            dic_.append(line)
    return dic_

RCTest_ = Relation_Chuck_test()

i = -1
j = -1

fw = open("test_chuck_5_triplet_llama2_13b_right_0.json", "w")
with open(triplet_KNN_top1_test) as fr:
    for line in fr.readlines():
        i = i + 1
        line = json.loads(line.strip())
        line_ = line["instruction"].split("Examples:")
        excamples = line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}", "")
        instruction_ = line_[0].strip()
        ex = RCTest_[i]
        
        sentence_ = ex.split("Response")[0]
        renew_exc = []
        if sentence_ not in excamples:
            j = j + 1
            ex = ex.strip().split("Context:")
            
            for excample in ex:
                if len(excample) != 0:
                    excample = excample.split("Response:")
                    input_ = excample[0]
                    response_ = excample[1].strip()
                    hts = R_HT_mapping[response_]
                    ht = random.sample(hts, 1)[0].split("|")
                    
                    renew_input = ht[0] + "" + input_ + "" + ht[1]
                    renew_response = ht[0] + "|" + response_ + "|" + ht[1]
                    
                    renew_exc.append("Context: " + renew_input + " Response: " + renew_response)
            
            instruction_ = instruction_ + " Examples: " + " ".join(renew_exc)
            line["instruction"] = instruction_
        
        fw.write(json.dumps(line))
        fw.write("\n")

fw.close()
print(f"✅ Generated test_chuck_5_triplet_llama2_13b_right_0.json with {i+1} examples")
