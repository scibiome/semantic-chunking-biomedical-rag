import json
import numpy as np

def load_data_base_relation_chuck():
    Chucks=[]
    relations=[]
    with open("stored_chucks_with_relation_5.txt","r") as fr:
        for line in fr.readlines():
            line=line.strip().split("--##--")
            Chucks.append(line[0])
            relations.append(line[1])
    return Chucks,relations

relation_chunks,relation_chunks_with_r=load_data_base_relation_chuck()
relation_chunks_embeddings=np.load("retrived_chucks_5.npy")

fw_=open("test_data_topn_chucks_5_deduplicated.json","w")
j=-1
with open("test_data_chucks_5_deduplicated.json","r") as fr:
    for line in fr.readlines():
        j=j+1
        if j % 100 == 0:
            print(f"Processing test example {j}")
        line=json.loads(line)
        
        sentence=line["sentence"].split(" ")
        all_chucks=line["all_chucks"]
        all_c_top_n={}
        for chucks in all_chucks:
            for ch, itssimchuk in chucks.items():
                for i in range(len(itssimchuk[0][:1])):
                    all_c_top_n[itssimchuk[0][i]]=itssimchuk[1][i]
        
        line["topn_sim_chuck"]=all_c_top_n
        fw_.write(json.dumps(line))
        fw_.flush()
        fw_.write("\n")

fw_.close()
print(f"✅ Completed test_data_topn_chucks_5.json with {j+1} examples")

