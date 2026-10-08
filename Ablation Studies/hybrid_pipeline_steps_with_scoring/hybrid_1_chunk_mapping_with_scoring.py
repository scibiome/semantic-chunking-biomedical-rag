import json
import numpy as np
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

# Add parent directory to path to import chunkers
sys.path.append(str(Path(__file__).parent.parent))

from configurable_semantic_chunker import ConfigurableSemanticChunker
from proposition_chunker import PropositionChunker

# Configuration
# Variant: Hybrid pipeline with tiered trigger and hierarchy scoring enabled
# This version applies tiered trigger bonuses and relation hierarchy to chunk scoring
# during candidate selection, combining with cosine similarity for final ranking.

MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'
DATASET_DIR = "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT"
CONFIG_FILE = "../configs/gmciht_config.json"
INPUT_FILE = f"{DATASET_DIR}/renew_triplet_T_dev.jsonl"
OUTPUT_FILE = "hybrid_gm-ciht_relation_with_chunks_5_with_scoring.json"
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
    # Ensure prerequisites
    if not os.path.exists(RELATION_EMBEDDING_FILE) or not os.path.exists(RELATION_DEF_FILE):
        print("Error: Run step 0 first to generate relation embeddings.")
        return

    # Initialize models
    model, tokenizer = init_model()
    
    # Initialize Chunkers
    print("Initializing Chunkers...")
    chunker = ConfigurableSemanticChunker(config_file=CONFIG_FILE, max_chunk_size=15)
    with open(CONFIG_FILE, 'r') as f:
        config_dict = json.load(f)
    prop_chunker = PropositionChunker(config_dict, entity_detector=chunker.detect_entities)
    
    # Load Relation Data
    relation_map = load_relation_map()
    relation_embeddings = np.load(RELATION_EMBEDDING_FILE)
    
    print(f"Processing {INPUT_FILE}...")
    
    relation_chunks_store = {}
    
    # Statistics to track source of selected chunks
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
                print(f"\n{'='*80}")
                print(f"Processing line {i}...")
                print(f"Current Stats:")
                for key, value in chunk_stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                print(f"{'='*80}\n")
                
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
            
            # 1. Smart Chunks (preserve metadata for tiered scoring)
            for c in chunker.get_all_candidates(sentence):
                if c['text'] not in seen_texts:
                    # Preserve chunk metadata (relation_type, trigger_word, etc.) for scoring
                    cand_dict = {'text': c['text'], 'source': 'smart'}
                    # Copy relevant metadata for _score_chunk
                    for key in ['relation_type', 'trigger_word', 'has_trigger', 
                               'is_uncertain', 'is_negated', 'context_type', 
                               'num_words', 'entities']:
                        if key in c:
                            cand_dict[key] = c[key]
                    candidates.append(cand_dict)
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
                    # Preserve proposition priority (includes tiered trigger bonus + hierarchy)
                    # Priority = relation_hierarchy_priority + tier_bonus (primary=3, secondary=2, tertiary=1)
                    # Convert priority to normalized score (0-10 range) for consistency with chunker scores
                    priority_score = p.priority / 2.0 if p.priority > 0 else 0.0  # Normalize priority to 0-10 range
                    candidates.append({
                        'text': p.text, 
                        'source': 'proposition',
                        'chunker_score': priority_score,  # Pre-calculated score from priority
                        'relation_type': p.relation_type,
                        'trigger': p.trigger,
                        'has_trigger': True
                    })
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

            # Score Candidates with Combined Scoring (Cosine Similarity + Tiered Trigger/Hierarchy Score)
            scored_chunks = []
            chunker_scores = []
            
            for cand in candidates:
                cand_text = cand['text']
                # FILTER: Skip very short chunks (single words or < 3 words)
                if len(cand_text.split()) < 3:
                    continue

                # Cosine similarity score (0-1 range, typically 0.3-0.9)
                cand_emb = make_text_embeddings(model, tokenizer, cand_text)
                sim = cos_similarity(rel_emb, cand_emb)
                
                # Chunker's tiered trigger/hierarchy score
                chunker_score = 0.0
                if 'chunker_score' in cand:
                    # Proposition chunks already have pre-calculated priority score
                    chunker_score = cand['chunker_score']
                elif cand.get('source') == 'smart':
                    # For configurable chunker chunks, use _score_chunk method
                    # This includes tiered trigger bonuses and hierarchy scoring
                    chunker_score = chunker._score_chunk(cand, relation)
                # Fixed chunks have no scoring (chunker_score remains 0.0)
                
                chunker_scores.append(chunker_score)
                
                # Normalize chunker score to 0-1 range for combination
                # Chunker scores are typically 0-10, normalize to 0-1
                normalized_chunker_score = chunker_score / 10.0 if chunker_score > 0 else 0.0
                
                # Combined score: 70% cosine similarity + 30% chunker tiered trigger/hierarchy score
                # This gives weight to both semantic similarity and trigger-based quality
                combined_score = 0.7 * sim + 0.3 * normalized_chunker_score
                
                scored_chunks.append((cand, sim, combined_score))
            
            # Track score statistics for analysis
            if chunker_scores:
                avg_chunker_score = sum(chunker_scores) / len(chunker_scores)
                max_chunker_score = max(chunker_scores)
                chunk_stats['avg_chunker_score'] = avg_chunker_score
                chunk_stats['max_chunker_score'] = max_chunker_score
                
                # Calculate combined score statistics
                combined_scores = [comb for _, _, comb in scored_chunks]
                if combined_scores:
                    avg_combined = sum(combined_scores) / len(combined_scores)
                    max_combined = max(combined_scores)
                    chunk_stats['avg_combined_score'] = avg_combined
                    chunk_stats['max_combined_score'] = max_combined
                
                # Calculate cosine similarity statistics
                cos_sims = [sim for _, sim, _ in scored_chunks]
                if cos_sims:
                    avg_cos_sim = sum(cos_sims) / len(cos_sims)
                    max_cos_sim = max(cos_sims)
                    chunk_stats['avg_cos_sim'] = avg_cos_sim
                    chunk_stats['max_cos_sim'] = max_cos_sim
            
            # Select Top 2 with Proposition Bias using COMBINED SCORE
            # Strategy:
            # 1. Sort all candidates by combined score (70% cosine + 30% tiered trigger/hierarchy)
            # 2. Check if any 'proposition' chunk is in the top 5
            # 3. If yes, and it's not in top 2, force-swap it into the second slot
            # 4. If no proposition chunk exists or it's low quality, just take top 2 by combined score
            
            scored_chunks.sort(key=lambda x: x[2], reverse=True)  # Sort by combined_score
            
            final_selection = []
            
            # Find best proposition chunk (using combined score)
            best_prop = None
            for cand, sim, combined in scored_chunks:
                if cand['source'] == 'proposition':
                    best_prop = (cand, sim, combined)
                    break
            
            if len(scored_chunks) > 0:
                # Always take the absolute best chunk (by combined score)
                # Store as (cand, sim) for compatibility with rest of code
                top_chunk = scored_chunks[0]
                final_selection.append((top_chunk[0], top_chunk[1]))
                
                # Log top chunk scores for monitoring (every 100 examples)
                if chunk_stats['total_processed'] % 100 == 0:
                    top_cand, top_sim, top_combined = top_chunk
                    # Get chunker score from candidate metadata or calculate from normalized
                    top_chunker_score = top_cand.get('chunker_score', 0.0)
                    if top_chunker_score == 0.0 and top_cand.get('source') == 'smart':
                        # Re-calculate to show the actual chunker score
                        top_chunker_score = chunker._score_chunk(top_cand, relation)
                    print(f"  [Sample {chunk_stats['total_processed']}] Top Chunk: {top_cand['source']}")
                    print(f"    Cosine Sim: {top_sim:.4f}, Chunker Score: {top_chunker_score:.2f}, Combined: {top_combined:.4f}")
                    print(f"    Text: {top_cand['text'][:80]}...")
                
                # For the second slot:
                if len(scored_chunks) > 1:
                    # If we found a proposition chunk and it wasn't the first one selected
                    if best_prop and best_prop[0] != scored_chunks[0][0]:
                        # Check if it's in the top 5 candidates overall (by combined score)
                        is_in_top_5 = False
                        for c, s, comb in scored_chunks[:5]:
                            if c['source'] == 'proposition':
                                is_in_top_5 = True
                                break
                        
                        if is_in_top_5:
                             # Use proposition chunk (store as cand, sim)
                             final_selection.append((best_prop[0], best_prop[1]))
                             # Log when proposition bias is applied
                             if chunk_stats['total_processed'] % 100 == 0:
                                 prop_cand, prop_sim, prop_combined = best_prop
                                 print(f"  [Sample {chunk_stats['total_processed']}] Prop Bias Applied: {prop_sim:.4f} cos, {prop_combined:.4f} combined")
                        else:
                             # If prop is really bad, don't force it, take 2nd best score
                             final_selection.append((scored_chunks[1][0], scored_chunks[1][1]))
                    else:
                        # No valid prop chunk to swap in, take 2nd best
                        final_selection.append((scored_chunks[1][0], scored_chunks[1][1]))

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

    # Save to file
    print("="*50)
    print("FINAL CHUNKING STATISTICS")
    # Convert numpy float types to regular Python floats for JSON serialization
    chunk_stats_serializable = {}
    for key, value in chunk_stats.items():
        if isinstance(value, (np.floating, np.integer)):
            chunk_stats_serializable[key] = float(value)
        elif isinstance(value, float):
            chunk_stats_serializable[key] = value
        else:
            chunk_stats_serializable[key] = value
    print(json.dumps(chunk_stats_serializable, indent=2))
    print("="*50)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as fw:
        for relation, chunks in relation_chunks_store.items():
             entry = {relation: list(chunks)}
             fw.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()

