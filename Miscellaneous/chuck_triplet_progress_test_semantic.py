import json
import random

# SEMANTIC CHUNKING: Use semantic INSTRUCTION files (from step 5)
relation_chuck_data_train = "../0_make_relation_chuck_and_scorer_data_semantic/train_chuck_instruction_5_llama13b_right_semantic.json"
relation_chuck_data_test = "../0_make_relation_chuck_and_scorer_data_semantic/test_chuck_instruction_5_llama13b_right_semantic.json"

triplet_KNN_top1_train = "0_Sunflower_triplet_extraction_13b/data/KNN_demo_train.json"
triplet_KNN_top1_test = "0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"

dev_source = "5_make_chuck/renew_triplet_T_dev.jsonl"

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

# SEMANTIC: Output to semantic-specific file
fw = open("test_chuck_5_triplet_llama2_13b_right_0_semantic.json", "w")
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
                if len(excample) != 0 and "Response:" in excample:
                    excample = excample.split("Response:")
                    input_ = excample[0]
                    response_ = excample[1].strip()
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
                    hts = R_HT_mapping[response_]
                    ht = random.sample(hts, 1)[0].split("|")
                    
                    renew_input = ht[0] + "" + input_ + "" + ht[1]
                    renew_response = ht[0] + "|" + response_ + "|" + ht[1]
                    renew_exc.append("Context: " + renew_input + " Response: " + renew_response)
            
            renew_exc = " ".join(renew_exc)
            new_instruction = instruction_ + " " + renew_exc
            line["instruction"] = new_instruction
            fw.write(json.dumps(line))
            fw.write("\n")
            fw.flush()
        
        else:
            line["instruction"] = line["instruction"].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}", "").replace("   ", " ")
            fw.write(json.dumps(line))
            fw.write("\n")
            fw.flush()

fw.close()
print(f"✅ Generated test_chuck_5_triplet_llama2_13b_right_0_semantic.json with {i+1} examples")




