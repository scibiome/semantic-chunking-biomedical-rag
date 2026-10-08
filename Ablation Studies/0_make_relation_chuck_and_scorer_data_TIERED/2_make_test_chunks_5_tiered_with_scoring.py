"""
Step 2b: Create Similar Chunks for Test Data (TIERED Variant - With Tiered Triggers + Scoring, No Hierarchy)
Variant that applies tiered trigger scoring during chunk creation in configurable chunker.
"""

import json
import numpy as np
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import chunkers (using chunker with scoring enabled)
from configurable_semantic_chunker_with_scoring import ConfigurableSemanticChunker
from proposition_chunker import PropositionChunker

# Configuration
MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'
DATASET_DIR = "../dataset/0_GM-CIHT"
INPUT_FILE = f"{DATASET_DIR}/renew_triplet_T_test_deduplicated.jsonl"
STORED_CHUNKS_FILE = "output/stored_chucks_with_relation_5_tiered.txt"
STORED_EMBEDDINGS_FILE = "output/retrived_chucks_5_tiered.npy"
OUTPUT_FILE = "output/test_data_chucks_5_tiered.json"
CONFIG_FILE = "configs/config.json"

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

def load_stored_chunks():
    chunks = []
    relations = []
    with open(STORED_CHUNKS_FILE, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                relations.append(parts[0])
                chunks.append(parts[1])
            else:
                relations.append("UNKNOWN")
                chunks.append(line.strip())
    return chunks, relations

def main():
    # Initialize model
    model, tokenizer = init_model()
    
    # Initialize Chunkers
    print("Initializing Chunkers...")
    chunker = ConfigurableSemanticChunker(config_file=CONFIG_FILE, max_chunk_size=15)
    with open(CONFIG_FILE, 'r') as f:
        config_dict = json.load(f)
    prop_chunker = PropositionChunker(config_dict, entity_detector=chunker.detect_entities)
    
    # Load stored chunks and embeddings
    print("Loading stored chunks...")
    stored_chunks, stored_relations = load_stored_chunks()
    stored_embeddings = np.load(STORED_EMBEDDINGS_FILE)
    
    print(f"Processing test data from {INPUT_FILE}...")
    
    with open(INPUT_FILE, "r") as fr, open(OUTPUT_FILE, "w") as fw:
        for j, line in enumerate(fr):
            if j % 50 == 0:
                print(f"Processing line {j}...")
                
            data = json.loads(line)
            sentence = data["SENTENCE"]
            # For test data, we might not use the ground truth relation for retrieval in a real scenario,
            # but usually we assume the relation is known or we retrieve for all relations.
            # In this pipeline, the input file has PREDICATE.
            relation = data.get("PREDICATE", "")
            subj_text = data.get("SUBJECT_TEXT", "")
            obj_text = data.get("OBJECT_TEXT", "")
            
            # Generate candidate chunks
            candidates = []
            seen_texts = set()
            
            # 1. Semantic chunks (with tiered trigger scoring)
            for c in chunker.get_all_candidates(sentence, relation=relation):
                if c['text'] not in seen_texts:
                    candidates.append(c['text'])
                    seen_texts.add(c['text'])
            
            # 2. Proposition chunks
            detected_ents = chunker.detect_entities(sentence)
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
            
            props = prop_chunker.extract_propositions(sentence, detected_ents, expected_relation=relation)
            for p in props:
                if p.text not in seen_texts:
                    candidates.append(p.text)
                    seen_texts.add(p.text)
            
            # 3. Fixed chunks
            words = sentence.split()
            splits = np.array_split(words, 5)
            for split in splits:
                if len(split) > 0:
                    text = " ".join(split)
                    if text not in seen_texts:
                        candidates.append(text)
                        seen_texts.add(text)
            
            # Find similar chunks
            all_chunks_data = []
            for chunk_text in candidates:
                chunk_emb = make_text_embeddings(model, tokenizer, chunk_text)
                sims = []
                for i in range(len(stored_embeddings)):
                    sim = cos_similarity(chunk_emb, stored_embeddings[i])
                    sims.append(sim)
                
                top_5_idx = np.argsort(sims)[-5:]
                top_5_chunks = [stored_chunks[i] for i in top_5_idx]
                top_5_rels = [stored_relations[i] for i in top_5_idx]
                
                chunk_sim_data = {}
                chunk_sim_data[chunk_text] = [top_5_chunks, top_5_rels]
                all_chunks_data.append(chunk_sim_data)
            
            final_dic = {
                "relation": relation,
                "head": subj_text,
                "tail": obj_text,
                "sentence": sentence,
                "all_chucks": all_chunks_data
            }
            fw.write(json.dumps(final_dic) + "\n")
            fw.flush()

    print(f"✅ Processing complete! Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

