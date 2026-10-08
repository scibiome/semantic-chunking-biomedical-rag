import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

# Configuration
MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'
INPUT_JSON = "hybrid_gm-ciht_relation_with_chunks_5.json"
OUTPUT_TXT = "hybrid_stored_chucks_with_relation_5.txt"
OUTPUT_NPY = "hybrid_retrived_chucks_5.npy"

def init_model():
    print(f"Loading model: {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    return model, tokenizer

def make_text_embeddings(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)
        hidden_states = output.hidden_states
        input_embeddings = hidden_states[0]
        embedding = torch.mean(input_embeddings.squeeze(0), dim=0)
        return embedding.detach().cpu().numpy()

def main():
    input_file = INPUT_JSON
    if not os.path.exists(input_file):
        # Fallback if user named it .txt in step 1 (I'll fix step 1 output name)
        if os.path.exists("stored_chucks_with_relation_5_improved.txt"):
             input_file = "stored_chucks_with_relation_5_improved.txt"
        else:
             print(f"Error: Input file {input_file} not found.")
             return

    model, tokenizer = init_model()
    
    all_chunks = []
    all_relations = []
    
    print("Reading chunks from step 1...")
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            for relation, chunks in data.items():
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_relations.append(relation)
    
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = []
    
    # Open text output file
    with open(OUTPUT_TXT, "w") as fw:
        for i, (chunk, relation) in enumerate(zip(all_chunks, all_relations)):
            if i % 100 == 0:
                print(f"Processed {i}/{len(all_chunks)}")
            
            # Save to text file: chunk --##-- relation
            fw.write(f"{chunk}--##--{relation}\n")
            
            # Compute embedding
            emb = make_text_embeddings(model, tokenizer, chunk)
            embeddings.append(emb)
            
    np.save(OUTPUT_NPY, np.array(embeddings))
    print(f"Saved text DB to {OUTPUT_TXT}")
    print(f"Saved embeddings DB to {OUTPUT_NPY}")

if __name__ == "__main__":
    main()

