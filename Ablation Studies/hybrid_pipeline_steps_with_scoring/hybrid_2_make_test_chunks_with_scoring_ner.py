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

# Import NER entity detector
from ner_entity_detector import create_biomedical_ner_detector

# Configuration
# Variant: Scoring pipeline with NER-based entity detection
MODEL_ID = '/home/users/riah/models/MedLLaMA_13B'
DATASET_DIR = "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT"
CONFIG_FILE = "../configs/gmciht_config.json"
INPUT_FILE = f"{DATASET_DIR}/renew_triplet_T_test_deduplicated.jsonl"
OUTPUT_FILE = "hybrid_test_data_chucks_5_with_scoring_ner.json"
DB_TXT = "hybrid_stored_chucks_with_relation_5_with_scoring_ner.txt"
DB_NPY = "hybrid_retrived_chucks_5_with_scoring_ner.npy"

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
            if cand_text in k_text:
                is_redundant = True
                break
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
    
    # Enable NER-based entity detection
    try:
        ner_detector = create_biomedical_ner_detector()
        chunker.detect_entities = ner_detector.detect_entities
        print("✅ NER-based entity detection enabled")
    except Exception as e:
        print(f"⚠️  Warning: Could not load NER system ({e})")
        print("   Falling back to regex-based entity detection")
    
    with open(CONFIG_FILE, 'r') as f:
        config_dict = json.load(f)
    prop_chunker = PropositionChunker(config_dict, entity_detector=chunker.detect_entities)
    
    # Load Database
    print("Loading vector database...")
    db_chunks, db_relations = load_database()
    db_embeddings = np.load(DB_NPY)
    
    print(f"Processing {INPUT_FILE}...")
    
    # Statistics tracking
    stats = {
        'total_processed': 0,
        'total_candidates': 0,
        'total_retrievals': 0,
        'avg_combined_scores': [],
        'avg_cosine_scores': [],
        'avg_chunker_scores': [],
        'tiered_trigger_usage': {'primary': 0, 'secondary': 0, 'tertiary': 0, 'none': 0},
        'source_distribution': {'smart': 0, 'proposition': 0, 'fixed': 0}
    }
    
    with open(INPUT_FILE, "r") as fr, open(OUTPUT_FILE, "w") as fw:
        for i, line in enumerate(fr):
            if i % 100 == 0:
                print(f"\n{'='*80}")
                print(f"Processing line {i}...")
                print(f"Stats so far:")
                print(f"  Total processed: {stats['total_processed']}")
                print(f"  Avg candidates per sentence: {stats['total_candidates']/max(stats['total_processed'], 1):.1f}")
                if stats['avg_combined_scores']:
                    print(f"  Avg combined score: {np.mean(stats['avg_combined_scores']):.4f}")
                    print(f"  Avg cosine score: {np.mean(stats['avg_cosine_scores']):.4f}")
                    print(f"  Avg chunker score: {np.mean(stats['avg_chunker_scores']):.4f}")
                print(f"{'='*80}\n")
            
            data = json.loads(line)
            sentence = data["SENTENCE"]
            relation = data.get("PREDICATE", None)  # Get relation for tiered scoring
            stats['total_processed'] += 1
            
            # --- HYBRID CHUNKING ---
            candidates = []
            seen = set()
            
            # 1. Smart chunks (preserve metadata for scoring)
            for c in chunker.get_all_candidates(sentence):
                 if c['text'] not in seen:
                     cand_dict = {'text': c['text'], 'source': 'smart'}
                     # Copy relevant metadata for _score_chunk (tiered triggers + hierarchy)
                     for key in ['relation_type', 'trigger_word', 'has_trigger', 
                                'is_uncertain', 'is_negated', 'context_type', 
                                'num_words', 'entities']:
                         if key in c:
                             cand_dict[key] = c[key]
                     candidates.append(cand_dict)
                     seen.add(c['text'])
                     stats['source_distribution']['smart'] += 1
            
            # 2. Proposition (with relation-aware filtering)
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
            
            # Pass expected_relation to filter propositions to only those matching the target relation
            props = prop_chunker.extract_propositions(sentence, detected_ents, expected_relation=relation)
            for p in props:
                if p.text not in seen:
                    # Store proposition metadata including priority (tiered triggers + hierarchy)
                    candidates.append({
                        'text': p.text, 
                        'source': 'proposition',
                        'chunker_score': float(p.priority) if hasattr(p, 'priority') and p.priority > 0 else 0.0,
                        'relation_type': p.relation_type if hasattr(p, 'relation_type') else None,
                        'trigger': p.trigger if hasattr(p, 'trigger') else None,
                        'has_trigger': True
                    })
                    seen.add(p.text)
                    stats['source_distribution']['proposition'] += 1
                    
                    # Track tiered trigger usage
                    if hasattr(p, 'trigger') and p.trigger and relation:
                        # Check which tier this trigger belongs to
                        tiers = chunker._get_tiered_triggers(relation) if hasattr(chunker, '_get_tiered_triggers') else {}
                        if p.trigger in tiers.get('primary', []):
                            stats['tiered_trigger_usage']['primary'] += 1
                        elif p.trigger in tiers.get('secondary', []):
                            stats['tiered_trigger_usage']['secondary'] += 1
                        elif p.trigger in tiers.get('tertiary', []):
                            stats['tiered_trigger_usage']['tertiary'] += 1
                        else:
                            stats['tiered_trigger_usage']['none'] += 1
            
            # 3. Fixed (Fallback)
            words = sentence.split()
            splits = np.array_split(words, 5)
            for split in splits:
                if len(split) > 0:
                    text = " ".join(split)
                    if text not in seen:
                        candidates.append({'text': text, 'source': 'fixed'})
                        seen.add(text)
                        stats['source_distribution']['fixed'] += 1
            
            stats['total_candidates'] += len(candidates)
            
            # Log chunk generation (every 100 examples)
            if i % 100 == 0:
                print(f"  [Line {i}] Generated {len(candidates)} candidates:")
                smart_count = sum(1 for c in candidates if c.get('source') == 'smart')
                prop_count = sum(1 for c in candidates if c.get('source') == 'proposition')
                fixed_count = sum(1 for c in candidates if c.get('source') == 'fixed')
                print(f"    Smart: {smart_count}, Proposition: {prop_count}, Fixed: {fixed_count}")
                if relation:
                    print(f"    Target relation: {relation}")
            
            # --- DEDUPLICATION ---
            # Now passing index 'i' for logging
            final_chunk_texts = filter_redundant_chunks(candidates, i)
            
            # --- RETRIEVAL WITH COMBINED SCORING ---
            # For each chunk, find neighbors using combined scoring (cosine + tiered triggers/hierarchy)
            all_chunks_data = []
            
            # Scoring constants (matching Step 1)
            COSINE_WEIGHT = 0.7
            CHUNKER_WEIGHT = 0.3
            MAX_CHUNKER_SCORE = 10.0  # Max score from _score_chunk
            
            for chunk_text in final_chunk_texts:
                if not chunk_text.strip():
                    continue
                    
                # FILTER: Skip very short chunks (single words or < 3 words)
                if len(chunk_text.split()) < 3:
                    continue
                    
                chunk_emb = make_text_embeddings(model, tokenizer, chunk_text)
                
                # Compute Combined Scores for all DB chunks
                # Strategy: For each DB chunk, compute:
                # 1. Cosine similarity (semantic)
                # 2. Chunker score (tiered triggers + hierarchy) - need to reconstruct metadata
                # 3. Combined score = 0.7 * cosine + 0.3 * normalized_chunker
                
                combined_scores = []
                chunker_scores_list = []
                cosine_scores_list = []
                
                for j, (db_chunk, db_rel) in enumerate(zip(db_chunks, db_relations)):
                    # Cosine similarity
                    sim = cos_similarity(chunk_emb, db_embeddings[j])
                    cosine_scores_list.append(sim)
                    
                    # Chunker score for DB chunk
                    # Reconstruct minimal metadata for scoring
                    db_chunk_meta = {
                        'text': db_chunk,
                        'relation_type': db_rel,
                        'num_words': len(db_chunk.split()),
                        'has_trigger': False,  # Will be detected by _score_chunk
                    }
                    
                    # Compute chunker score (includes tiered triggers + hierarchy)
                    db_chunker_score = chunker._score_chunk(db_chunk_meta, relation)
                    chunker_scores_list.append(db_chunker_score)
                    normalized_db_chunker = min(db_chunker_score / MAX_CHUNKER_SCORE, 1.0)
                    
                    # Combined score: 70% cosine + 30% chunker (tiered triggers + hierarchy)
                    combined_score = COSINE_WEIGHT * sim + CHUNKER_WEIGHT * normalized_db_chunker
                    combined_scores.append((j, sim, db_chunker_score, combined_score, db_chunk, db_rel))
                
                # Track statistics
                if combined_scores:
                    stats['avg_combined_scores'].append(np.mean([cs[3] for cs in combined_scores]))
                    stats['avg_cosine_scores'].append(np.mean(cosine_scores_list))
                    stats['avg_chunker_scores'].append(np.mean(chunker_scores_list))
                
                # Sort by combined score (descending)
                combined_scores.sort(key=lambda x: x[3], reverse=True)
                
                # Log retrieval details (every 100 examples)
                if i % 100 == 0:
                    print(f"  [Line {i}] Retrieval for chunk: {chunk_text[:60]}...")
                    print(f"    Top 5 DB chunks by COMBINED SCORE:")
                    for rank, (idx, cos_sim, chunker_sc, comb_sc, db_chunk, db_rel) in enumerate(combined_scores[:5], 1):
                        # Detect tier from chunker score (if it has trigger bonuses)
                        tier_info = ""
                        if chunker_sc > 5.0:  # High score suggests primary trigger
                            tier_info = " [likely primary trigger]"
                        elif chunker_sc > 3.0:
                            tier_info = " [likely secondary trigger]"
                        elif chunker_sc > 1.0:
                            tier_info = " [likely tertiary trigger]"
                        
                        print(f"      {rank}. Cosine: {cos_sim:.4f}, Chunker: {chunker_sc:.2f}, Combined: {comb_sc:.4f}{tier_info}")
                        print(f"         DB chunk: {db_chunk[:70]}...")
                        print(f"         DB relation: {db_rel}")
                    
                    # Show comparison: top by cosine vs top by combined
                    if len(combined_scores) > 0:
                        # Sort by cosine for comparison
                        cosine_sorted = sorted(combined_scores, key=lambda x: x[1], reverse=True)
                        top_by_cosine = cosine_sorted[0]
                        top_by_combined = combined_scores[0]
                        
                        if top_by_cosine[0] != top_by_combined[0]:
                            print(f"    ⚠️  Ranking difference:")
                            print(f"      Top by COSINE: {top_by_cosine[4][:60]}... (cos={top_by_cosine[1]:.4f}, combined={top_by_cosine[3]:.4f})")
                            print(f"      Top by COMBINED: {top_by_combined[4][:60]}... (cos={top_by_combined[1]:.4f}, combined={top_by_combined[3]:.4f})")
                            print(f"      → Combined scoring changed ranking (tiered triggers/hierarchy influence)")
                
                # Top 5 by combined score
                top_5_idx = [idx for idx, _, _, _, _, _ in combined_scores[:5]]
                
                top_5_chunks = [db_chunks[idx] for idx in top_5_idx]
                top_5_rels = [db_relations[idx] for idx in top_5_idx]
                
                stats['total_retrievals'] += 1
                
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

    # Final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Total sentences processed: {stats['total_processed']}")
    print(f"Total candidates generated: {stats['total_candidates']}")
    print(f"Average candidates per sentence: {stats['total_candidates']/max(stats['total_processed'], 1):.2f}")
    print(f"Total retrievals performed: {stats['total_retrievals']}")
    
    if stats['avg_combined_scores']:
        print(f"\nScoring Statistics:")
        print(f"  Average combined score: {np.mean(stats['avg_combined_scores']):.4f} (std: {np.std(stats['avg_combined_scores']):.4f})")
        print(f"  Average cosine score: {np.mean(stats['avg_cosine_scores']):.4f} (std: {np.std(stats['avg_cosine_scores']):.4f})")
        print(f"  Average chunker score: {np.mean(stats['avg_chunker_scores']):.4f} (std: {np.std(stats['avg_chunker_scores']):.4f})")
        print(f"  Max combined score: {np.max(stats['avg_combined_scores']):.4f}")
        print(f"  Min combined score: {np.min(stats['avg_combined_scores']):.4f}")
    
    print(f"\nSource Distribution:")
    for source, count in stats['source_distribution'].items():
        percentage = (count / max(stats['total_candidates'], 1)) * 100
        print(f"  {source.capitalize()}: {count} ({percentage:.1f}%)")
    
    print(f"\nTiered Trigger Usage in Propositions:")
    total_tiered = sum(stats['tiered_trigger_usage'].values())
    if total_tiered > 0:
        for tier, count in stats['tiered_trigger_usage'].items():
            percentage = (count / total_tiered) * 100
            print(f"  {tier.capitalize()}: {count} ({percentage:.1f}%)")
    else:
        print("  No tiered trigger data available")
    
    print("="*80)
    
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
