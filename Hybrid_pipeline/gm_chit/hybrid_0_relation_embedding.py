import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

# Configuration
MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'
OUTPUT_FILE = "relation_description.npy"
INPUT_FILE = "relation_with_defination.txt"

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
        # Mean pooling
        embedding = torch.mean(input_embeddings.squeeze(0), dim=0)
        return embedding.detach().cpu().numpy()

def main():
    if not os.path.exists(INPUT_FILE):
        # Try copying from parent directory if not found
        if os.path.exists(f"../{INPUT_FILE}"):
            print(f"Copying {INPUT_FILE} from parent directory...")
            os.system(f"cp ../{INPUT_FILE} .")
        else:
            print(f"Error: {INPUT_FILE} not found.")
            return

    model, tokenizer = init_model()
    
    embeddings = []
    print("Generating relation embeddings...")
    
    with open(INPUT_FILE, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                # relation = parts[0]
                description = parts[1]
                emb = make_text_embeddings(model, tokenizer, description)
                embeddings.append(emb)
    
    np.save(OUTPUT_FILE, np.array(embeddings))
    print(f"Saved embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

