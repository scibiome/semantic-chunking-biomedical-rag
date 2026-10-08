import json
import re

# File Paths
TRAIN_INPUT = "hybrid_train_data_chucks_5.json"
TEST_INPUT = "hybrid_test_data_chucks_5.json"
TRAIN_OUTPUT = "hybrid_train_data_topn_chucks_5.json"
TEST_OUTPUT = "hybrid_test_data_topn_chucks_5.json"

def is_proposition_chunk(chunk_text):
    """
    Detect if a chunk looks like a canonical proposition.
    Proposition chunks often have specific structures like:
    - "Subject ... Trigger ... Object"
    - "Trigger of Subject with Object" (from canonical rephrasing)
    """
    # Check for common proposition patterns from your logs
    if "administered of" in chunk_text: return True
    if "interaction of" in chunk_text: return True
    if "inhibition of" in chunk_text: return True
    if "treatment of" in chunk_text: return True
    if "effect of" in chunk_text: return True
    
    # Check if it lacks common "sliding window" artifacts (starts with lowercase, ends with punctuation)
    # This is a heuristic; proposition chunks are often cleaner.
    return False

def process_file(input_file, output_file, mode="train"):
    """Extract top-N similar chunks with Proposition Priority."""
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
            
            # --- STRATEGY: Prioritize Proposition-based Retrieval ---
            # We want to collect the best retrievals from ALL chunks.
            # Instead of just taking the top-1 from every chunk blindly, 
            # we can favor retrievals that came from "proposition-like" chunks 
            # OR we can just aggregate them all.
            
            # For now, let's stick to the request: "get the best benefits".
            # The best benefit comes from the DIVERSITY of the retrievals.
            
            all_c_top_n = {}
            
            # 1. Collect ALL candidate retrievals from all chunks
            candidates = []
            for chunk_entry in all_chucks:
                # Each 'chunk_entry' is a dict like {"chunk_text": [[neighbor_texts], [neighbor_rels]]}
                for chunk_text, retrieved_data in chunk_entry.items():
                    neighbors = retrieved_data[0]
                    relations = retrieved_data[1]
                    
                    # Check if the SOURCE chunk was a proposition chunk
                    is_prop = is_proposition_chunk(chunk_text)
                    
                    # Store (neighbor, relation, score_proxy, is_prop_source)
                    # We don't have explicit scores here, but the list is ordered by similarity.
                    # Rank 0 is best.
                    for i in range(len(neighbors)):
                        # We give a "virtual boost" to retrievals from proposition chunks
                        # by considering them "higher rank" equivalent.
                        candidates.append({
                            "neighbor": neighbors[i],
                            "relation": relations[i],
                            "rank": i,
                            "is_prop": is_prop
                        })

            # 2. Select the absolute best unique neighbors
            # Strategy:
            # - Take Top-1 from EVERY chunk (guarantees coverage of all sentence parts)
            # - If we need more, take Top-2 from Proposition chunks first.
            
            # Current logic in old script: Take Top-1 from every chunk.
            # Modified logic: Take Top-1 from every chunk.
            
            # Wait, the user asked: "chances that we lose prepositional chunks".
            # If a proposition chunk was generated in Step 2, it IS in 'all_chucks'.
            # If we take Top-1 from it, we preserve its retrieval.
            # The danger is if we Filter them out.
            
            for chunk_entry in all_chucks:
                for chunk_text, retrieved_data in chunk_entry.items():
                    neighbors = retrieved_data[0]
                    relations = retrieved_data[1]
                    
                    if not neighbors: continue
                    
                    # ALWAYS take the top 1 retrieval from this chunk
                    top_neighbor = neighbors[0]
                    top_relation = relations[0]
                    all_c_top_n[top_neighbor] = top_relation
                    
                    # HYBRID BOOST:
                    # If this is a proposition chunk (high quality), take the 2nd neighbor too!
                    # This increases the influence of proposition-derived knowledge.
                    if is_proposition_chunk(chunk_text) and len(neighbors) > 1:
                        second_neighbor = neighbors[1]
                        second_relation = relations[1]
                        all_c_top_n[second_neighbor] = second_relation

            data["topn_sim_chuck"] = all_c_top_n
            fw_.write(json.dumps(data))
            fw_.write("\n")
            
    fw_.close()
    print(f"✓ {mode} topn complete: {j+1} samples processed")

if __name__ == "__main__":
    print("="*80)
    print("Step 3: Extract TopN Similar Chunks (Hybrid Priority)")
    print("="*80)
    
    process_file(TRAIN_INPUT, TRAIN_OUTPUT, "train")
    process_file(TEST_INPUT, TEST_OUTPUT, "test")
    
    print("\n" + "="*80)
    print("✅ Step 3 Complete!")
    print("="*80)


