import json
import re
import math

# File Paths
# Variant: Scoring pipeline - uses chunks from tiered trigger/hierarchy scoring
TRAIN_INPUT = "hybrid_train_data_chucks_5_with_scoring_ner.json"
TEST_INPUT = "hybrid_test_data_chucks_5_with_scoring_ner.json"
TRAIN_OUTPUT = "hybrid_train_data_topn_scored_5_with_scoring_ner.json"
TEST_OUTPUT = "hybrid_test_data_topn_scored_5_with_scoring_ner.json"

# Trigger words for relations (simplified list)
TRIGGERS = {
    'inhibits': ['inhibit', 'block', 'suppress', 'prevent', 'reduce', 'decrease', 'attenuate', 'abolish'],
    'stimulates': ['stimulate', 'activate', 'induce', 'increase', 'enhance', 'promote', 'elevate', 'upregulat'],
    'treats': ['treat', 'therap', 'cures', 'alleviate', 'remedy', 'heal', 'manag'],
    'prevents': ['prevent', 'prophyla', 'protect', 'avoid'],
    'produces': ['produce', 'generat', 'yield', 'synthesi', 'create'],
    'interacts': ['interact', 'bind', 'associat', 'complex', 'link'],
    'causes': ['cause', 'lead to', 'result in', 'trigger', 'provoke']
}

def has_trigger(text):
    """Check if text contains any relation trigger."""
    text_lower = text.lower()
    for category, words in TRIGGERS.items():
        for word in words:
            if word in text_lower:
                return True
    return False

def is_proposition_chunk(chunk_text):
    """
    Detect if a chunk looks like a canonical proposition.
    1. Look for explicit [Head] / {Tail} markers (from Proposition Chunker)
    2. Look for canonical sentence structures
    """
    # 1. Explicit Markers (Strongest Signal)
    if '[' in chunk_text and ']' in chunk_text and '{' in chunk_text and '}' in chunk_text:
        return True
        
    # 2. Canonical Patterns
    if "administered of" in chunk_text: return True
    if "interaction of" in chunk_text: return True
    if "inhibition of" in chunk_text: return True
    if "treatment of" in chunk_text: return True
    if "effect of" in chunk_text: return True
    
    return False

def calculate_quality_score(text, rank):
    """
    Calculate a semantic quality score for a retrieved chunk.
    Higher is better.
    """
    score = 0.0
    
    # 1. Base Score from Rank (0=Best)
    # Decay factor: 1.0 for rank 0, 0.9 for rank 1, etc.
    score += 1.0 / (rank + 1.0)
    
    # 2. Proposition Bonus (+2.0)
    # If the retrieved text ITSELF looks like a proposition
    if is_proposition_chunk(text):
        score += 2.0
        
    # 3. Trigger Bonus (+1.0)
    if has_trigger(text):
        score += 1.0
        
    # 4. Length Penalty
    # We want "Goldilocks" length: 5-25 words.
    words = text.split()
    word_count = len(words)
    
    if word_count < 5:
        score -= 0.5 # Too short context
    elif word_count > 30:
        score -= 1.0 # Too long/noisy
    elif word_count > 20:
        score -= 0.2 # Slightly long
        
    # 5. Boundary Penalty
    # Starts/Ends with preposition/conjunction often indicates split context
    bad_starts = ['and', 'but', 'or', 'of', 'in', 'with', 'to', 'for']
    if words[0].lower() in bad_starts:
        score -= 0.3
        
    return score

def process_file(input_file, output_file, mode="train"):
    """Extract top-N chunks using Semantic Quality Scoring."""
    print(f"Processing {mode} data from {input_file}...")
    
    fw_ = open(output_file, "w")
    
    j = -1
    with open(input_file, "r") as fr:
        for line in fr.readlines():
            j += 1
            if j % 100 == 0:
                print(f"Processed {j} {mode} samples...")
            
            data = json.loads(line)
            all_chucks = data.get("all_chucks", [])
            
            # --- NEW STRATEGY: Semantic Quality Selection ---
            candidates = []
            seen_texts = set()
            
            # 1. Collect ALL candidates from ALL source chunks
            for chunk_entry in all_chucks:
                for chunk_text, retrieved_data in chunk_entry.items():
                    neighbors = retrieved_data[0]
                    relations = retrieved_data[1]
                    
                    # Original Chunk Boost:
                    # If the SOURCE chunk was a proposition, its retrievals are likely better contextually
                    source_is_prop = is_proposition_chunk(chunk_text)
                    
                    for i in range(len(neighbors)):
                        neighbor_text = neighbors[i]
                        relation = relations[i]
                        
                        if neighbor_text in seen_texts:
                            continue
                            
                        # Calculate Score
                        quality_score = calculate_quality_score(neighbor_text, i)
                        
                        # Apply boost from Source
                        if source_is_prop:
                            quality_score += 0.5
                            
                        candidates.append({
                            'text': neighbor_text,
                            'relation': relation,
                            'score': quality_score
                        })
                        seen_texts.add(neighbor_text)
            
            # 2. Sort by Score (Descending)
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 3. Select Top-N (e.g., Top 15 to ensure Step 5 has enough options)
            top_n_candidates = candidates[:15]
            
            # 4. Convert to format expected by pipeline: dict {text: relation}
            # Note: This loses the order/score, but the downstream script (Step 5)
            # typically re-scores or just takes keys.
            # CRITICAL: We must ensure Step 5 preserves this order or we re-sort there.
            # Assuming standard python dicts preserve insertion order (Python 3.7+),
            # inserting them in sorted order helps.
            
            all_c_top_n = {}
            for cand in top_n_candidates:
                all_c_top_n[cand['text']] = cand['relation']

            data["topn_sim_chuck"] = all_c_top_n
            fw_.write(json.dumps(data))
            fw_.write("\n")
            
    fw_.close()
    print(f"✓ {mode} topn complete: {j+1} samples processed")

if __name__ == "__main__":
    print("="*80)
    print("Step 3: Extract TopN Similar Chunks (Semantic Scored)")
    print("="*80)
    
    process_file(TRAIN_INPUT, TRAIN_OUTPUT, "train")
    process_file(TEST_INPUT, TEST_OUTPUT, "test")
    
    print("\n" + "="*80)
    print("✅ Step 3 Complete!")
    print("="*80)
