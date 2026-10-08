"""
GM-CIHT improved pipeline - Step 2a: Make similar chunks for TRAIN data.
Reads stored chunks + embeddings from step 1b, chunks train sentences, writes train_data_chucks_5_improved.json.
Prerequisite: step 1b (stored_chucks_with_relation_5_improved.txt, retrived_chucks_5_improved.npy).
"""
import json
import numpy as np
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import hybrid semantic chunker
sys.path.insert(0, os.path.dirname(__file__))
from improved_semantic_chunker import ImprovedSemanticChunker

_script_dir = os.path.dirname(os.path.abspath(__file__))
model_id = '/home/users/riah/models/MedLLaMA_13B'
# model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
# tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# Initialize improved semantic chunker
print("Initializing Improved Semantic Chunker for train data...")
semantic_chunker = ImprovedSemanticChunker(
    window_size=7,
    stride=3,
    min_chunk_size=5,
    max_chunk_size=15
)
print("Improved Chunker ready!")



def cos_similisty(vector1,vector2):
    import numpy as np

    vector1 = np.array(vector1)

    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)

    norm_vector1 = np.linalg.norm(vector1)

    norm_vector2 = np.linalg.norm(vector2)

    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

    return cosine_similarity



def make_text_embeddings(description):
    input_ids = tokenizer.encode(description, return_tensors='pt', truncation=True, max_length=512)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)
        hidden_states = output.hidden_states
        input_embeddings = hidden_states[0]
        input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)
        final_embeddings = input_embeddings.detach().cpu().numpy()
    return final_embeddings

def load_data_base_relation_chuck():
    Chucks = []
    relations = []
    stored_path = os.path.join(_script_dir, "stored_chucks_with_relation_5_improved.txt")
    with open(stored_path, "r") as fr:
        for line in fr.readlines():
            line = line.strip().split("--##--")
            Chucks.append(line[0])
            relations.append(line[1])
    return Chucks, relations

relation_chunks, relation_chunks_with_r = load_data_base_relation_chuck()
_emb_path = os.path.join(_script_dir, "retrived_chucks_5_improved.npy")
if not os.path.isfile(_emb_path):
    raise FileNotFoundError(f"Step 1b output not found: {_emb_path}. Run step 1b first.")
relation_chunks_embeddings = np.load(_emb_path)


def make_train_test_chunck(one_chuck_n_words):
    output_path = os.path.join(_script_dir, "train_data_chucks_5_improved.json")
    train_input = os.path.join(_script_dir, "..", "dataset", "0_GM-CIHT", "renew_triplet_T_train.jsonl")
    fw_ = open(output_path, "w")

    relation_chucks = {}
    j = -1
    with open(train_input, "r") as fr:
        for line in fr.readlines():
            j = j + 1
            if j >= 0:
                if (j + 1) % 100 == 0:
                    print(j + 1, flush=True)
                line=json.loads(line)

                # Get sentence and ground truth relation
                sentence_str = line["SENTENCE"]
                gt_relation = line.get("PREDICATE", None)  # Get ground truth relation
                
                # IMPROVED SEMANTIC CHUNKING: Entity + relation aware
                try:
                    chunks = semantic_chunker.chunk_for_triplet_extraction(
                        sentence_str,
                        relation=gt_relation,  # Use ground truth relation for relation-aware chunking
                        num_chunks=one_chuck_n_words
                    )
                    # Extract text from chunk dictionaries
                    chunked_data = [c['text'] for c in chunks]
                except Exception as e:
                    print(f"Warning: Improved chunking failed for sentence {j}: {e}")
                    # Fallback to sentence
                    chunked_data = [sentence_str]
                    
                chucned_data_new=[]
                all_chucks=[]
                for chuck in chunked_data:
                    cos_=[]
                    relations=[]

                    # chuck is already a string from hybrid chunker
                    chucned_data_new.append(chuck)
                    chuck_embedding=make_text_embeddings(chuck)
                    i=-1
                    for cu in relation_chunks:
                        i=i+1
                        store_chuck_vector=relation_chunks_embeddings[i]
                        cos_.append(cos_similisty(chuck_embedding,store_chuck_vector))
                    
    
                    top_5_idx = np.argsort(cos_)[-5:]

                    top_5_chucks = [relation_chunks[i] for i in top_5_idx]
                    top_5_relations = [relation_chunks_with_r[i] for i in top_5_idx]
                    
                    chuck_sim_chucks={}
                    chuck_sim_chucks[chuck]=[top_5_chucks,top_5_relations]
                    all_chucks.append(chuck_sim_chucks)
                # print(all_chucks)
                final_dic={}
                final_dic["relation"]=line["PREDICATE"]
                final_dic["head"]=line["SUBJECT_TEXT"]
                final_dic["tail"]=line["OBJECT_TEXT"]
                final_dic["sentence"]=sentence_str  # Use original sentence string
                final_dic["all_chucks"]=all_chucks
                fw_.write(json.dumps(final_dic))
                fw_.flush()
                fw_.write("\n")

    fw_.close()
    print(f"\n✅ Step 2a (Train) completed successfully!")
    print(f"Processed {j+1} training sentences")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    one_chuck_n_words=5
    make_train_test_chunck(one_chuck_n_words)
