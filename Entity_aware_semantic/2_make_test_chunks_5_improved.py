import json
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import sys
import os

# Import improved semantic chunker
sys.path.insert(0, os.path.dirname(__file__))
from improved_semantic_chunker import ImprovedSemanticChunker

model_id = '/home/users/riah/models/MedLLaMA_13B'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, torch_dtype=torch.float16, device_map='auto',
)

tokenizer = LlamaTokenizer.from_pretrained(model_id)

# Initialize improved semantic chunker
print("Initializing Improved Semantic Chunker for test data...")
semantic_chunker = ImprovedSemanticChunker(
    window_size=7,
    stride=3,
    min_chunk_size=5,
    max_chunk_size=15
)
print("Improved Chunker ready!")


def cos_similisty(vector1, vector2):
    import numpy as np
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return cosine_similarity


def make_text_embeddings(discription):
    input_ids = tokenizer.encode(discription, return_tensors='pt')
    output = model(input_ids, output_hidden_states=True)
    hidden_states = output.hidden_states
    input_embeddings = hidden_states[0]
    input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)
    final_embeddings = input_embeddings.detach().cpu().numpy()
    return final_embeddings


def load_data_base_relation_chuck():
    Chucks = []
    relations = []
    with open("stored_chucks_with_relation_5_improved.txt", "r") as fr:
        for line in fr.readlines():
            line = line.strip().split("--##--")
            Chucks.append(line[0])
            relations.append(line[1])
    return Chucks, relations


relation_chunks, relation_chunks_with_r = load_data_base_relation_chuck()
relation_chunks_embeddings = np.load("retrived_chucks_5_improved.npy")


def make_test_chunks(one_chuck_n_words):
    # Use deduplicated test file and rename output to indicate deduplicated source
    fw_ = open("test_data_chucks_5_improved_deduplicated.json", "w")

    j = -1
    with open("../dataset/0_GM-CIHT/renew_triplet_T_test_deduplicated.jsonl", "r") as fr:
        for line in fr.readlines():
            j = j + 1
            if j >= 0:
                print(f"Processing test sample {j}")
                line = json.loads(line)

                # Get sentence as string
                sentence_str = line["SENTENCE"]
                
                # IMPROVED SEMANTIC CHUNKING
                try:
                    # Get ground truth relation for relation-aware chunking
                    gt_relation = line.get("PREDICATE", None)
                    chunks = semantic_chunker.chunk_for_triplet_extraction(
                        sentence_str,
                        relation=gt_relation,
                        num_chunks=one_chuck_n_words
                    )
                    # Extract text from chunk dictionaries
                    chunked_data = [c['text'] for c in chunks]
                except Exception as e:
                    print(f"Warning: Improved chunking failed for test sample {j}: {e}")
                    chunked_data = [sentence_str]

                all_chucks = []
                for chuck in chunked_data:
                    cos_ = []
                    # chuck is already a string from improved chunker
                    chuck_embedding = make_text_embeddings(chuck)
                    for i, _ in enumerate(relation_chunks):
                        store_chuck_vector = relation_chunks_embeddings[i]
                        cos_.append(cos_similisty(chuck_embedding, store_chuck_vector))

                    top_5_idx = np.argsort(cos_)[-5:]
                    top_5_chucks = [relation_chunks[i] for i in top_5_idx]
                    top_5_relations = [relation_chunks_with_r[i] for i in top_5_idx]

                    chuck_sim_chucks = {chuck: [top_5_chucks, top_5_relations]}
                    all_chucks.append(chuck_sim_chucks)

                final_dic = {
                    "relation": line["PREDICATE"],
                    "head": line["SUBJECT_TEXT"],
                    "tail": line["OBJECT_TEXT"],
                    "sentence": sentence_str,  # Use original sentence string
                    "all_chucks": all_chucks,
                }
                fw_.write(json.dumps(final_dic))
                fw_.flush()
                fw_.write("\n")


if __name__ == "__main__":
    one_chuck_n_words = 5
    make_test_chunks(one_chuck_n_words)
