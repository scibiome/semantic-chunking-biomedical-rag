"""
Step 3b: Make Top-N Chunks for Test Data (Semantic + Proposition Variant)
With Semantic Quality Scoring
"""

import json
import numpy as np
import sys
import os

# Configuration
INPUT_FILE = "output/test_data_chucks_5_semantic_prop.json"
OUTPUT_FILE = "output/test_data_topn_chucks_5_semantic_prop.json"

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

def make_topn_for_test():
    print(f"Processing {INPUT_FILE}...")
    fw_ = open(OUTPUT_FILE, "w")
    
    j = -1
    with open(INPUT_FILE, "r") as fr:
        for line in fr.readlines():
            j = j + 1
            if j % 100 == 0:
                print(f"Processing line {j}...")
                
            data = json.loads(line.strip())
            all_chucks = data.get("all_chucks", [])
            
            # --- NEW STRATEGY: Semantic Quality Selection ---
            candidates = []
            seen_texts = set()
            
            # 1. Collect ALL candidates from ALL source chunks
            for chunk_entry in all_chucks:
                # Handle list of dicts structure: [{'chunk_text': [[sims], [rels]]}, ...]
                if isinstance(chunk_entry, dict):
                    for chunk_text, retrieved_data in chunk_entry.items():
                        neighbors = retrieved_data[0]
                        relations = retrieved_data[1]
                        
                        # Original Chunk Boost
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
            
            # 3. Select Top-N (e.g., Top 15)
            # Typically for instruction creation we usually just need the best few,
            # but standard pipeline usually preserves top 1 or mapped structure.
            # In Hybrid pipeline it stored top 15.
            top_n_candidates = candidates[:15]
            
            # 4. Convert to dict {text: relation}
            # Note: Preserving order by inserting into dict
            all_c_top_n = {}
            for cand in top_n_candidates:
                all_c_top_n[cand['text']] = cand['relation']
            
            data["topn_sim_chuck"] = all_c_top_n
            
            # Remove heavy 'all_chucks'
            if "all_chucks" in data:
                del data["all_chucks"]
                
            fw_.write(json.dumps(data))
            fw_.write("\n")
            fw_.flush()
            
    fw_.close()
    print(f"✅ Processing complete! Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    make_topn_for_test()
