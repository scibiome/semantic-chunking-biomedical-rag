import json
import numpy as np


## Note: Model loading not needed here; we only reorganize precomputed similar chunks

def load_data_base_relation_chuck():
    Chucks=[]
    relations=[]
    with open("stored_chucks_with_relation_5_ddi.txt","r") as fr:
        for line in fr.readlines():
            line=line.strip().split("--##--")
            Chucks.append(line[0])
            relations.append(line[1])

    return Chucks,relations

relation_chunks,relation_chunks_with_r=load_data_base_relation_chuck()
relation_chunks_embeddings=np.load("retrived_chucks_5_ddi.npy")


def make_topn_for_train():
    """Extract top-N similar chunks for training data"""
    fw_=open("train_data_topn_chucks_5_ddi.json","w")

    j=-1
    with open("train_data_chucks_5_ddi.json","r") as fr:
        for line in fr.readlines():
            j=j+1
            print(f"Processing train example {j}", flush=True)
            line=json.loads(line)

            sentence=line["sentence"].split(" ")
            all_chucks=line["all_chucks"]  # one sentence many chucks
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
    print(f"✅ Completed train_data_topn_chucks_5_ddi.json with {j+1} examples")


def make_topn_for_test():
    """Extract top-N similar chunks for test data"""
    fw_=open("test_data_topn_chucks_5_ddi.json","w")

    j=-1
    with open("test_data_chucks_5_ddi.json","r") as fr:
        for line in fr.readlines():
            j=j+1
            print(f"Processing test example {j}", flush=True)
            line=json.loads(line)

            sentence=line["sentence"].split(" ")
            all_chucks=line["all_chucks"]  # one sentence many chucks
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
    print(f"✅ Completed test_data_topn_chucks_5_ddi.json with {j+1} examples")


if __name__=="__main__":
    print("Step 3: Extracting top-N similar chunks for train and test")
    print("="*60)
    
    # Process train data
    print("\nProcessing TRAIN data...")
    make_topn_for_train()
    
    # Process test data
    print("\nProcessing TEST data...")
    make_topn_for_test()
    
    print("\n✅ Step 3 completed successfully!")
