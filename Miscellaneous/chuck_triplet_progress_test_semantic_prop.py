import json
import random

# SEMANTIC + PROPOSITION CHUNKING (Test Data)
relation_chuck_data_test="../0_make_relation_chuck_and_scorer_data_SEMANTIC_PROP/output/test_chuck_instruction_5_semantic_prop.json"

triplet_KNN_top1_test="0_Sunflower_triplet_extraction_13b/data/KNN_demo_test.json"
dev_source="5_make_chuck/renew_triplet_T_dev.jsonl" # Still needed for HT_R mapping? Or should use train?

# For test generation, we often just need to replace the examples in the instruction
# but keep the input/target from the test set.

# Wait, in the training script, we replace EXAMPLES in the prompt with retrieved chunks.
# Same logic applies to test.

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

def Relation_Chuck_test():
    dic_=[]
    with open(relation_chuck_data_test) as fr:
        for line in fr.readlines():
            # Extract retrieved examples
            # Format in instruction file: "instruction": "... Examples: ..."
            line=json.loads(line.strip())["instruction"].split("Examples:")[1]
            dic_.append(line)
    return dic_

RCTest_=Relation_Chuck_test()

i=-1
j=-1

fw=open("test_triplet_chunk_scorer_semantic_prop_s_hrt.json","w")
with open(triplet_KNN_top1_test) as fr:
        for line in fr.readlines():
            i=i+1
            line=json.loads(line.strip())
            line_=line["instruction"].split("Examples:")
            
            # The placeholder text might be different in test file?
            # Usually it's the same prompt template.
            excamples=line_[1].replace("1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}","")
            instruction_=line_[0].strip()
            
            if i >= len(RCTest_):
                 print(f"Warning: Index {i} out of bounds for test instruction chunks (len={len(RCTest_)})")
                 continue

            ex=RCTest_[i]

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
                        
                        response_parts = response_.split("|")
                        if len(response_parts) >= 2:
                            response_ = response_parts[1].strip()
                        else:
                            continue
                            
                        if not response_ or response_ in ['{}', '{}.']:
                            continue
                        
                        # In Test script, do we re-sample the entities?
                        # The original script does HT_R replacement.
                        # This creates synthetic examples from retrieved chunks.
                        if response_ in R_HT_mapping:
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
                line["instruction"] = line["instruction"].replace(
                    "1. Context:The effect of bicarbonate on liver alcohol  dehydrogenase. Response: bicarbonate|INTERACTS_WITH|alcohol. 2. Context:{}. Response: {}",
                    ""
                ).replace("   ", " ")
                fw.write(json.dumps(line))
                fw.write("\n")
                fw.flush()

fw.close()
print(f"Test samples processed: {j}")
print(f"Output file: test_triplet_chunk_scorer_semantic_prop_s_hrt.json")
