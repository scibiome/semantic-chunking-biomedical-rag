"""
Step 1: Generate Semantic + Proposition Chunks (No Tiered, No Hierarchy)
"""

import json
import numpy as np
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

# Import chunkers from local directory
from configurable_semantic_chunker import ConfigurableSemanticChunker
from proposition_chunker import PropositionChunker

# Configuration
MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'
DATASET_DIR = "../dataset/0_GM-CIHT"
INPUT_FILE = f"{DATASET_DIR}/renew_triplet_T_dev.jsonl"
OUTPUT_RELATION_FILE = "output/relation_with_chucks_5_semantic_prop.json"
OUTPUT_STORED_FILE = "output/stored_chucks_with_relation_5_semantic_prop.txt"
CONFIG_FILE = "configs/config.json"
RELATION_EMBEDDING_FILE = "relation_description.npy"
RELATION_DEF_FILE = "relation_with_defination.txt"

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

def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def load_relation_map():
    rd = {}
    with open(RELATION_DEF_FILE, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            rd[parts[0]] = i
    return rd

def main():
    # Ensure output dir
    os.makedirs(os.path.dirname(OUTPUT_RELATION_FILE), exist_ok=True)
    
    # Initialize model
    model, tokenizer = init_model()
    
    # Initialize Chunkers
    print("Initializing Chunkers...")
    chunker = ConfigurableSemanticChunker(config_file=CONFIG_FILE, max_chunk_size=15)
    
    # Load config for prop chunker
    with open(CONFIG_FILE, 'r') as f:
        config_dict = json.load(f)
        
    prop_chunker = PropositionChunker(config_dict, entity_detector=chunker.detect_entities)
    
    print(f"Config loaded: {config_dict['description']}")
    print(f"Tiered Triggers: {config_dict.get('use_tiered_triggers', False)}")
    print(f"Relation Hierarchy: {config_dict.get('use_relation_hierarchy', False)}")
    
    # Load Relation Data
    relation_map = load_relation_map()
    relation_embeddings = np.load(RELATION_EMBEDDING_FILE)
    
    print(f"Processing {INPUT_FILE}...")
    
    relation_chunks_store = {}
    stored_chunks_with_relation = []
    all_chunk_texts = set()
    
    chunk_stats = {
        'total_processed': 0,
        'smart_selected': 0,
        'proposition_selected': 0,
        'fixed_selected': 0,
        'total_candidates': 0
    }
    
    with open(INPUT_FILE, "r") as f:
        for i, line in enumerate(f):
            if i % 100 == 0:
                print(f"Processing line {i}...")
                print(f"Current Stats: {json.dumps(chunk_stats, indent=2)}")
                
            data = json.loads(line)
            sentence = data["SENTENCE"]
            relation = data["PREDICATE"]
            subj_text = data.get("SUBJECT_TEXT", "")
            obj_text = data.get("OBJECT_TEXT", "")
            
            if relation not in relation_map:
                continue
            
            chunk_stats['total_processed'] += 1    
            rel_emb = relation_embeddings[relation_map[relation]]
            
            # --- HYBRID CHUNKING STRATEGY ---
            candidates = []
            seen_texts = set()
            
            # 1. Smart Chunks
            for c in chunker.get_all_candidates(sentence):
                if c['text'] not in seen_texts:
                    candidates.append({'text': c['text'], 'source': 'smart'})
                    seen_texts.add(c['text'])
            
            # 2. Proposition Chunks
            # Combine regex entities with Ground Truth entities
            detected_ents = chunker.detect_entities(sentence)
            
            # Add Ground Truth entities
            for ent_text in [subj_text, obj_text]:
                if ent_text:
                    start = 0
                    while True:
                        idx = sentence.lower().find(ent_text.lower(), start)
                        if idx == -1: break
                        
                        # Check overlap with existing
                        is_new = True
                        new_span = (idx, idx+len(ent_text))
                        for e in detected_ents:
                            if max(idx, e['start']) < min(idx+len(ent_text), e['end']):
                                is_new = False
                                break
                        
                        if is_new:
                            detected_ents.append({
                                'text': sentence[idx:idx+len(ent_text)],
                                'start': idx,
                                'end': idx+len(ent_text)
                            })
                        start = idx + 1
            
            props = prop_chunker.extract_propositions(sentence, detected_ents, expected_relation=relation)
            for p in props:
                if p.text not in seen_texts:
                    candidates.append({'text': p.text, 'source': 'proposition'})
                    seen_texts.add(p.text)
            
            # 3. Fixed Chunks (Fallback/Baseline)
            words = sentence.split()
            splits = np.array_split(words, 5) # Default 5 chunks like original
            for split in splits:
                if len(split) > 0:
                    text = " ".join(split)
                    if text not in seen_texts:
                        candidates.append({'text': text, 'source': 'fixed'})
                        seen_texts.add(text)
            
            chunk_stats['total_candidates'] += len(candidates)

            # Score Candidates
            scored_chunks = []
            for cand in candidates:
                cand_text = cand['text']
                # FILTER: Skip very short chunks (single words or < 3 words)
                if len(cand_text.split()) < 3:
                    continue

                cand_emb = make_text_embeddings(model, tokenizer, cand_text)
                sim = cos_similarity(rel_emb, cand_emb)
                scored_chunks.append((cand, sim))
            
            # Select Top 2 with Proposition Bias
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            final_selection = []
            
            # Find best proposition chunk
            best_prop = None
            for cand, score in scored_chunks:
                if cand['source'] == 'proposition':
                    best_prop = (cand, score)
                    break
            
            if len(scored_chunks) > 0:
                # Always take the absolute best chunk
                final_selection.append(scored_chunks[0])
                
                # For the second slot:
                if len(scored_chunks) > 1:
                    # If we found a proposition chunk and it wasn't the first one selected
                    if best_prop and best_prop[0] != scored_chunks[0][0]:
                        # Check if it's in the top 5 candidates overall
                        is_in_top_5 = False
                        for c, s in scored_chunks[:5]:
                            if c['source'] == 'proposition':
                                is_in_top_5 = True
                                break
                        
                        if is_in_top_5:
                             final_selection.append(best_prop)
                        else:
                             # If prop is really bad, don't force it, take 2nd best score
                             final_selection.append(scored_chunks[1])
                    else:
                        # No valid prop chunk to swap in, take 2nd best
                        final_selection.append(scored_chunks[1])

            top_chunks_data = [x[0] for x in final_selection]
            top_chunks_text = [x['text'] for x in top_chunks_data]
            
            # Update stats
            for chunk_data in top_chunks_data:
                if chunk_data['source'] == 'smart':
                    chunk_stats['smart_selected'] += 1
                elif chunk_data['source'] == 'proposition':
                    chunk_stats['proposition_selected'] += 1
                elif chunk_data['source'] == 'fixed':
                    chunk_stats['fixed_selected'] += 1

            if relation not in relation_chunks_store:
                relation_chunks_store[relation] = set()
            relation_chunks_store[relation].update(top_chunks_text)
            
            # Store chunks for step 2 (similarity matching)
            for chunk_text in top_chunks_text:
                if chunk_text not in all_chunk_texts:
                    stored_chunks_with_relation.append(f"{relation}\t{chunk_text}")
                    all_chunk_texts.add(chunk_text)

    # Save relation chunks
    print(f"Saving to {OUTPUT_RELATION_FILE}...")
    with open(OUTPUT_RELATION_FILE, "w") as fw:
        for relation, chunks in relation_chunks_store.items():
             entry = {relation: list(chunks)}
             fw.write(json.dumps(entry) + "\n")
             
    # Save stored chunks for step 2
    print(f"Saving stored chunks to {OUTPUT_STORED_FILE}...")
    with open(OUTPUT_STORED_FILE, "w") as fw:
        for line in stored_chunks_with_relation:
            fw.write(line + "\n")
            
    print("="*50)
    print("FINAL CHUNKING STATISTICS")
    print(json.dumps(chunk_stats, indent=2))
    print("="*50)

if __name__ == "__main__":
    main()

