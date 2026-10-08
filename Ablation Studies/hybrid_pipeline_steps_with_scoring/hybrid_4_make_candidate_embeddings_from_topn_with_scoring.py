#!/usr/bin/env python3
"""
Generate embeddings for candidate chunks extracted from Top-N selections.
This creates embeddings for the high-quality chunks that were actually selected,
rather than the entire candidate pool.
"""

import json
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np
import os
import sys
import sqlite3

# File Paths
# Input: Candidate chunks file extracted from Top-N
# Variant: Scoring pipeline - uses chunks from tiered trigger/hierarchy scoring
CANDIDATE_CHUNKS_FILE = "hybrid_all_candiate_chunks_5_from_topn_with_scoring.txt"

# Output files (with _from_topn suffix to distinguish)
OUTPUT_PREFIX = "hybrid_all_posible_chuck_vector_5_from_topn_with_scoring_shard_"
TEXT_OUTPUT = "hybrid_all_candiate_chunks_5_from_topn_with_scoring_deduplicated.txt"

MODEL_ID = '/home/users/riah/models/Llama-2-13b-hf'

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

    tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
    
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

def _stream_candidates_from_file():
    """Generator that yields candidate strings from the extracted chunks file.
    Each line is a chunk text, we format it as 'Context: <chunk> Response: <relation>'
    But since we don't have relation info in the file, we'll just use the chunk text.
    """
    if not os.path.exists(CANDIDATE_CHUNKS_FILE):
        raise FileNotFoundError(f"Candidate chunks file not found: {CANDIDATE_CHUNKS_FILE}")
    
    print(f"Streaming candidates from {CANDIDATE_CHUNKS_FILE}...")
    with open(CANDIDATE_CHUNKS_FILE, "r", encoding='utf-8') as f:
        for line in f:
            chunk_text = line.strip()
            if not chunk_text:
                continue
            
            # Format as "Context: <chunk>" (matching original format)
            # The original script used "Context: <chunk> Response: <relation>"
            # But we only have chunk text, so we'll use just the chunk
            yield chunk_text

def _embed_batch(model, tokenizer, texts):
    """Generate embeddings for a batch of texts using first hidden state (embedding layer)."""
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
        # Use first hidden state (embedding layer) to match original script
        hs = output.hidden_states[0]
        
        # Mean pooling
        attn = (input_ids != pad_id).unsqueeze(-1).to(hs.dtype)
        summed = (hs * attn).sum(dim=1)
        lengths = attn.sum(dim=1).clamp(min=1)
        mean = summed / lengths
        return mean.detach().cpu().numpy()

def main():
    print("="*80)
    print("Step 4: Generate Candidate Embeddings from Top-N Selected Chunks")
    print("="*80)
    
    # Check if input file exists
    if not os.path.exists(CANDIDATE_CHUNKS_FILE):
        print(f"❌ Error: {CANDIDATE_CHUNKS_FILE} not found!")
        print(f"   Please run extract_candidates_from_topn.py first.")
        sys.exit(1)
    
    # Clean up old deduplication DB if exists
    db_file = "hybrid_candidates_from_topn_with_scoring_seen_deduplicated.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    model, tokenizer = setup_model()
    
    fw = open(TEXT_OUTPUT, "w", encoding='utf-8')
    
    # On-disk dedup to preserve global set() semantics without OOM
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS seen (text TEXT PRIMARY KEY)")
    conn.commit()

    shard_index = 0
    batch_size = 16  # Small batch size to match original safe behavior
    total = 0
    batch = []
    
    print("Starting stream processing...")
    
    try:
        for cand in _stream_candidates_from_file():
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

    print("\n" + "="*80)
    print(f"✅ Step 4 Complete! Total unique candidates: {total}")
    print(f"Generated {shard_index} shards.")
    print(f"Text output: {TEXT_OUTPUT}")
    print(f"Embedding shards: {OUTPUT_PREFIX}*.npy")
    print("="*80)

if __name__ == "__main__":
    main()






"""
Generate embeddings for candidate chunks extracted from Top-N selections.
This creates embeddings for the high-quality chunks that were actually selected,
rather than the entire candidate pool.
"""

import json
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np
import os
import sys
import sqlite3

# File Paths
# Input: Candidate chunks file extracted from Top-N
# Variant: Scoring pipeline - uses chunks from tiered trigger/hierarchy scoring
CANDIDATE_CHUNKS_FILE = "hybrid_all_candiate_chunks_5_from_topn_with_scoring.txt"

# Output files (with _from_topn suffix to distinguish)
OUTPUT_PREFIX = "hybrid_all_posible_chuck_vector_5_from_topn_with_scoring_shard_"
TEXT_OUTPUT = "hybrid_all_candiate_chunks_5_from_topn_with_scoring_deduplicated.txt"

MODEL_ID = '/home/users/riah/models/Llama-2-13b-hf'

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

    tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
    
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

def _stream_candidates_from_file():
    """Generator that yields candidate strings from the extracted chunks file.
    Each line is a chunk text, we format it as 'Context: <chunk> Response: <relation>'
    But since we don't have relation info in the file, we'll just use the chunk text.
    """
    if not os.path.exists(CANDIDATE_CHUNKS_FILE):
        raise FileNotFoundError(f"Candidate chunks file not found: {CANDIDATE_CHUNKS_FILE}")
    
    print(f"Streaming candidates from {CANDIDATE_CHUNKS_FILE}...")
    with open(CANDIDATE_CHUNKS_FILE, "r", encoding='utf-8') as f:
        for line in f:
            chunk_text = line.strip()
            if not chunk_text:
                continue
            
            # Format as "Context: <chunk>" (matching original format)
            # The original script used "Context: <chunk> Response: <relation>"
            # But we only have chunk text, so we'll use just the chunk
            yield chunk_text

def _embed_batch(model, tokenizer, texts):
    """Generate embeddings for a batch of texts using first hidden state (embedding layer)."""
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
        # Use first hidden state (embedding layer) to match original script
        hs = output.hidden_states[0]
        
        # Mean pooling
        attn = (input_ids != pad_id).unsqueeze(-1).to(hs.dtype)
        summed = (hs * attn).sum(dim=1)
        lengths = attn.sum(dim=1).clamp(min=1)
        mean = summed / lengths
        return mean.detach().cpu().numpy()

def main():
    print("="*80)
    print("Step 4: Generate Candidate Embeddings from Top-N Selected Chunks")
    print("="*80)
    
    # Check if input file exists
    if not os.path.exists(CANDIDATE_CHUNKS_FILE):
        print(f"❌ Error: {CANDIDATE_CHUNKS_FILE} not found!")
        print(f"   Please run extract_candidates_from_topn.py first.")
        sys.exit(1)
    
    # Clean up old deduplication DB if exists
    db_file = "hybrid_candidates_from_topn_with_scoring_seen_deduplicated.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    model, tokenizer = setup_model()
    
    fw = open(TEXT_OUTPUT, "w", encoding='utf-8')
    
    # On-disk dedup to preserve global set() semantics without OOM
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS seen (text TEXT PRIMARY KEY)")
    conn.commit()

    shard_index = 0
    batch_size = 16  # Small batch size to match original safe behavior
    total = 0
    batch = []
    
    print("Starting stream processing...")
    
    try:
        for cand in _stream_candidates_from_file():
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

    print("\n" + "="*80)
    print(f"✅ Step 4 Complete! Total unique candidates: {total}")
    print(f"Generated {shard_index} shards.")
    print(f"Text output: {TEXT_OUTPUT}")
    print(f"Embedding shards: {OUTPUT_PREFIX}*.npy")
    print("="*80)

if __name__ == "__main__":
    main()











