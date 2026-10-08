"""
Step 1b: Generate embeddings for stored chunks (Semantic + Proposition Variant)
"""

import numpy as np
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration
MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'
INPUT_FILE = "output/stored_chucks_with_relation_5_semantic_prop.txt"
OUTPUT_FILE = "output/retrived_chucks_5_semantic_prop.npy"

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
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return

    # Initialize model
    model, tokenizer = init_model()
    
    print(f"Processing {INPUT_FILE}...")
    
    chunks = []
    chunk_embeddings = []
    
    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()
        total_lines = len(lines)
        print(f"Total chunks to process: {total_lines}")
        
        for i, line in enumerate(lines):
            if i % 100 == 0:
                print(f"Processing chunk {i}/{total_lines}...")
            
            # Format: relation\tchunk_text
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
                
            chunk_text = parts[1]
            embedding = make_text_embeddings(model, tokenizer, chunk_text)
            chunk_embeddings.append(embedding)
            
    # Save embeddings
    print(f"Saving embeddings to {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, np.array(chunk_embeddings))
    print("✅ Done!")

if __name__ == "__main__":
    main()

