"""
Step 4: Create Candidate Embeddings (TIERED Variant)
Following hybrid script structure with permutations, batch processing, and deduplication
"""

import json
from itertools import permutations, islice
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import os
import sys
import sqlite3

# File Paths (Updated for tiered variant)
TRAIN_INPUT = "output/train_data_topn_chucks_5_tiered.json"
TEST_INPUT = "output/test_data_topn_chucks_5_tiered.json"
# Create separate directory for shards
SHARDS_DIR = "output/embeddings_shards"
# We will use a unified prefix for all shards to match hybrid behavior
OUTPUT_PREFIX = os.path.join(SHARDS_DIR, "all_posible_chuck_vector_5_tiered_shard_")
TEXT_OUTPUT = "output/all_candiate_chunks_5_tiered.txt"

MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'

def setup_model():
    """Initialize model and tokenizer with 4-bit quantization"""
    print(f"Loading model: {MODEL_ID}")
    
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
    
    # Robust pad token handling
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '</s>', 'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'

    # Resolve pad_token_id
    if tokenizer.pad_token_id is None:
        pid = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        if pid is None or pid == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            pid = tokenizer.convert_tokens_to_ids('[PAD]')
        tokenizer.pad_token_id = pid

    # Make model aware of PAD
    try:
        model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
        
    return model, tokenizer

def _permutation_strings(items, max_total=64):
    """Yield up to max_total permutations per record.
    Matches hybrid logic:
    - Always include the first full-length permutation
    - Then include shorter-length permutations in increasing length order
    """
    n = len(items)
    emitted = 0

    # First full-length permutation
    if n > 0:
        first_full = next(permutations(items, n), None)
        if first_full:
            yield " ".join(first_full)
            emitted += 1
            if emitted >= max_total:
                return

    # Then shorter permutations starting from length 1 upwards
    for i in range(1, n):
        for p in permutations(items, i):
            if not p:
                continue
            yield " ".join(p)
            emitted += 1
            if emitted >= max_total:
                return

def _stream_candidates():
    """Generator that yields candidate strings one by one."""
    
    # 1. Process Train and Test JSONs (matching hybrid pipeline for fair comparison)
    input_files = []
    if os.path.exists(TRAIN_INPUT): 
        input_files.append(TRAIN_INPUT)
    if os.path.exists(TEST_INPUT): 
        input_files.append(TEST_INPUT)
    
    for src in input_files:
        print(f"Streaming candidates from {src}...")
        with open(src, "r") as fr:
            for line in fr:
                try:
                    data = json.loads(line)
                    topn_sim_chuck = data.get("topn_sim_chuck", {})
                    items = []
                    for key, value in topn_sim_chuck.items():
                        items.append("Context: " + key + " Response: " + value)
                    
                    # Generate permutations lazily
                    for s in _permutation_strings(items):
                        yield s
                except json.JSONDecodeError:
                    continue

    # 2. Add KNN context/instruction (matching hybrid pipeline for fair comparison)
    knn_files = [
        "../2_relation_data_to_triple_train_data/KNN_demo_train.json",
        "../2_relation_data_to_triple_train_data/KNN_demo_test.json"
    ]
    
    for knn in knn_files:
        if os.path.exists(knn):
            print(f"Streaming KNN candidates from {knn}...")
            with open(knn, "r") as fr:
                for line in fr:
                    try:
                        data = json.loads(line)
                        context = data.get("context", "")
                        instr_field = data.get("instruction", "")
                        if "Examples:  " in instr_field:
                            instruction = instr_field.split("Examples:  ", 1)[1]
                        else:
                            instruction = instr_field
                        
                        if instruction: yield instruction
                        if context: yield context
                    except json.JSONDecodeError:
                        continue

def _embed_batch(model, tokenizer, texts):
    """Generate embeddings for a batch of texts using last hidden state logic."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Tokenize
    encoded = [tokenizer.encode(t, add_special_tokens=True, truncation=True, max_length=256) for t in texts]
    if not encoded:
        return np.zeros((0, getattr(model.config, 'hidden_size', 5120)), dtype=np.float32)
        
    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=model.device)
    
    for i, ids in enumerate(encoded):
        if len(ids) > 0:
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)

    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)
        # Use hidden_states[0] (embedding layer) to match hybrid script logic
        hs = output.hidden_states[0]
        
        # Mean pooling
        attn = (input_ids != pad_id).unsqueeze(-1).to(hs.dtype)
        summed = (hs * attn).sum(dim=1)
        lengths = attn.sum(dim=1).clamp(min=1)
        mean = summed / lengths
        return mean.detach().cpu().numpy()

def main():
    print("="*80)
    print("Step 4: Generate Candidate Embeddings (TIERED - Optimized Stream)")
    print("="*80)
    
    # Ensure shards directory exists
    os.makedirs(SHARDS_DIR, exist_ok=True)
    print(f"Shards will be saved to: {SHARDS_DIR}")
    
    if os.path.exists("tiered_candidates_seen_deduplicated.db"):
        os.remove("tiered_candidates_seen_deduplicated.db")
        
    model, tokenizer = setup_model()
    
    fw = open(TEXT_OUTPUT, "w")
    
    # On-disk dedup to preserve global set() semantics without OOM
    conn = sqlite3.connect("tiered_candidates_seen_deduplicated.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS seen (text TEXT PRIMARY KEY)")
    conn.commit()

    shard_index = 0
    batch_size = 16  # Small batch size to match hybrid safe behavior
    total = 0
    batch = []
    
    print("Starting stream processing...")
    
    try:
        for cand in _stream_candidates():
            # Deduplicate
            try:
                cur.execute("INSERT INTO seen(text) VALUES (?)", (cand,))
                conn.commit()
            except sqlite3.IntegrityError:
                continue # Skip duplicate
            
            # Save text
            fw.write(cand)
            fw.write("\n")
            total += 1
            batch.append(cand)
            
            # Embed if batch full
            if len(batch) >= batch_size:
                embeds = _embed_batch(model, tokenizer, batch)
                fname = f"{OUTPUT_PREFIX}{shard_index:05d}.npy"
                np.save(fname, embeds)
                shard_index += 1
                batch.clear()
                
                if total % 1000 == 0:
                    print(f"Processed {total} unique candidates...")
        
        # Process remaining
        if batch:
            embeds = _embed_batch(model, tokenizer, batch)
            fname = f"{OUTPUT_PREFIX}{shard_index:05d}.npy"
            np.save(fname, embeds)
            shard_index += 1
            batch.clear()
            
    finally:
        fw.close()
        conn.close()
        # Cleanup DB if desired, or keep for debugging
        # os.remove("tiered_candidates_seen_deduplicated.db")

    print("\n" + "="*80)
    print(f"✅ Step 4 Complete! Total unique candidates: {total}")
    print(f"Generated {shard_index} shards.")
    print("="*80)

if __name__ == "__main__":
    main()
