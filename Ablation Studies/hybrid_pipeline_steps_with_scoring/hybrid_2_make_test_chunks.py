import json
import numpy as np
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configurable_semantic_chunker import ConfigurableSemanticChunker
from proposition_chunker import PropositionChunker

# Configuration
MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'
DATASET_DIR = "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT"
CONFIG_FILE = "../configs/gmciht_config.json"
INPUT_FILE = f"{DATASET_DIR}/renew_triplet_T_test_deduplicated.jsonl" # Assuming test set
OUTPUT_FILE = "hybrid_test_data_chucks_5_with_scoring.json"
DB_TXT = "hybrid_stored_chucks_with_relation_5_with_scoring.txt"
DB_NPY = "hybrid_retrived_chucks_5_with_scoring.npy"

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

def load_database():
    chunks = []
    relations = []
    with open(DB_TXT, "r") as f:
        for line in f:
            parts = line.strip().split("--##--")
            if len(parts) == 2:
                chunks.append(parts[0])
                relations.append(parts[1])
    return chunks, relations

def filter_redundant_chunks(candidates_list, i=-1):
    """
    Removes chunks that are substrings of other chunks or highly similar.
    Prioritizes PROPOSITION > SMART > FIXED.
    Within same priority, prioritizes SHORTER chunks to reduce noise,
    as long as they contain the core info (which Prop/Smart usually do).
    
    candidates_list: list of dicts {'text': str, 'source': str}
    """
    # Define Source Priority (Lower is better)
    priority_map = {'proposition': 0, 'smart': 1, 'fixed': 2}
    
    # Sort by: 1. Priority (Asc), 2. Length (Asc) - prefer shorter, cleaner chunks
    sorted_cands = sorted(candidates_list, key=lambda x: (priority_map.get(x['source'], 3), len(x['text'])))
    
    kept = []
    
    for cand in sorted_cands:
        cand_text = cand['text']
        cand_words = set(cand_text.lower().split())
        if not cand_words: continue

        is_redundant = False
        
        for k in kept:
            k_text = k['text']
            k_words = set(k_text.lower().split())
            
            # 1. Substring Check
            # If current candidate is substring of kept (unlikely given sort order prefer shorter, but possible if diff priority)
            if cand_text in k_text:
                is_redundant = True
                break
            # If kept is substring of current candidate (e.g. kept is Prop "drug treats X", current is Fixed "The drug treats X")
            # We want to discard the longer "noisy" one.
            if k_text in cand_text:
                is_redundant = True
                break
                
            # 2. High Overlap Check (Jaccard > 0.8)
            intersection = len(cand_words.intersection(k_words))
            union = len(cand_words.union(k_words))
            if union > 0 and (intersection / union) > 0.8:
                is_redundant = True
                break
        
        if not is_redundant:
            kept.append(cand)
            
    # LOGGING (Sampled)
    if i != -1 and i % 100 == 0:
        print(f"\n--- Deduplication Log (Line {i}) ---")
        print("Candidates:")
        for c in candidates_list:
            print(f"  [{c['source']}] {c['text']}")
        print("Selected:")
        for k in kept:
            print(f"  [{k['source']}] {k['text']}")
        print("------------------------------------\n")

    return [k['text'] for k in kept]

def main():
    if not os.path.exists(DB_TXT) or not os.path.exists(DB_NPY):
        print("Error: Database files (txt/npy) not found. Run Step 1b.")
        return

    model, tokenizer = init_model()
    
    # Initialize Chunkers
    chunker = ConfigurableSemanticChunker(config_file=CONFIG_FILE, max_chunk_size=15)
    with open(CONFIG_FILE, 'r') as f:
        config_dict = json.load(f)
    prop_chunker = PropositionChunker(config_dict, entity_detector=chunker.detect_entities)
    
    # Load Database
    print("Loading vector database...")
    db_chunks, db_relations = load_database()
    db_embeddings = np.load(DB_NPY)
    
    print(f"Processing {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as fr, open(OUTPUT_FILE, "w") as fw:
        for i, line in enumerate(fr):
            if i % 100 == 0:
                print(f"Processing line {i}...")
            
            data = json.loads(line)
            sentence = data["SENTENCE"]
            
            # --- HYBRID CHUNKING ---
            candidates = []
            seen = set()
            
            # 1. Smart
            for c in chunker.get_all_candidates(sentence):
                 if c['text'] not in seen:
                     candidates.append({'text': c['text'], 'source': 'smart'})
                     seen.add(c['text'])
            
            # 2. Proposition
            # Augment entities with ground truth (Test set usually has entities provided for RE task)
            detected_ents = chunker.detect_entities(sentence)
            subj_text = data.get("SUBJECT_TEXT", "")
            obj_text = data.get("OBJECT_TEXT", "")
            
            for ent_text in [subj_text, obj_text]:
                if ent_text:
                    start = 0
                    while True:
                        idx = sentence.lower().find(ent_text.lower(), start)
                        if idx == -1: break
                        is_new = True
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
            
            props = prop_chunker.extract_propositions(sentence, detected_ents)
            for p in props:
                if p.text not in seen:
                    candidates.append({'text': p.text, 'source': 'proposition'})
                    seen.add(p.text)
            
            # 3. Fixed (Fallback)
            words = sentence.split()
            splits = np.array_split(words, 5)
            for split in splits:
                if len(split) > 0:
                    text = " ".join(split)
                    if text not in seen:
                        candidates.append({'text': text, 'source': 'fixed'})
                        seen.add(text)
            
            # --- DEDUPLICATION ---
            # Now passing index 'i' for logging
            final_chunk_texts = filter_redundant_chunks(candidates, i)
            
            # --- RETRIEVAL ---
            # For each chunk, find neighbors
            all_chunks_data = []
            
            for chunk_text in final_chunk_texts:
                if not chunk_text.strip():
                    continue
                    
                # FILTER: Skip very short chunks (single words or < 3 words)
                if len(chunk_text.split()) < 3:
                    continue
                    
                chunk_emb = make_text_embeddings(model, tokenizer, chunk_text)
                
                scores = []
                for j, db_emb in enumerate(db_embeddings):
                    sim = cos_similarity(chunk_emb, db_emb)
                    scores.append(sim)
                
                # Top 5
                top_5_idx = np.argsort(scores)[-5:]
                
                top_5_chunks = [db_chunks[idx] for idx in top_5_idx]
                top_5_rels = [db_relations[idx] for idx in top_5_idx]
                
                # Format: {chunk_text: [ [neighbors], [neighbor_relations] ]}
                chunk_data = {chunk_text: [top_5_chunks, top_5_rels]}
                all_chunks_data.append(chunk_data)
            
            # Final Output
            final_dic = {
                "relation": data["PREDICATE"],
                "head": data["SUBJECT_TEXT"],
                "tail": data["OBJECT_TEXT"],
                "sentence": sentence,
                "all_chucks": all_chunks_data
            }
            
            fw.write(json.dumps(final_dic) + "\n")
            fw.flush()

    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
