# Step 0.1: Generate Relation Embeddings for Semantic Chunking
# This is the same as fixed-size chunking since relation definitions don't change

import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = '/home/users/riah/models/MedLLaMA_13B'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

def relation_with_defination():
    RD = {}
    with open("relation_with_defination.txt", "r") as fr:
        i = -1
        for line in fr.readlines():
            i = i + 1
            line = line.strip().split("\t")
            RD[line[0]] = i
    return RD

R_D = relation_with_defination()

def make_text_embeddings(description):
    input_ids = tokenizer.encode(description, return_tensors='pt')
    output = model(input_ids, output_hidden_states=True)
    hidden_states = output.hidden_states
    input_embeddings = hidden_states[0]
    input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)
    final_embeddings = input_embeddings.detach().cpu().numpy()
    return final_embeddings

print("Generating relation embeddings...")
all_vectors = []
relation_names = []

with open("relation_with_defination.txt", "r") as fr:
    for line in fr.readlines():
        line = line.strip().split("\t")
        relation_name = line[0]
        relation_description = line[1]
        
        print(f"Processing relation: {relation_name}")
        relation_embedding = make_text_embeddings(relation_description)
        all_vectors.append(relation_embedding)
        relation_names.append(relation_name)

# Save relation embeddings
all_vectors = np.array(all_vectors)
np.save("relation_description_improved.npy", all_vectors)

print(f"✅ Generated {len(all_vectors)} relation embeddings")
print(f"✅ Saved to: relation_description_improved.npy")
print(f"✅ Embedding shape: {all_vectors.shape}")

