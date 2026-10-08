import json
import re
from itertools import permutations

# Define file paths
KNN_TRAIN = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_train.json"
KNN_TEST = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_test.json"

HYBRID_TRAIN_INPUT = "hybrid_train_data_topn_scored_5.json"
HYBRID_TEST_INPUT = "hybrid_test_data_topn_scored_5.json"

OUTPUT_TRAIN = "hybrid_train_instruction_5_permuted_smart_scored.json"
OUTPUT_TEST = "hybrid_test_instruction_5_permuted_smart_scored.json"

def _read_topn_lines(topn_path):
    with open(topn_path, "r") as fr:
        for raw in fr:
            yield json.loads(raw)

def parse_chunk_triplet(chunk_text, relation):
    """
    Extracts [Head] and {Tail} if present, or uses robust fallback.
    Returns: (clean_text, response, is_propositional)
    """
    is_prop = False
    
    # Check for Proposition Chunker signatures (canonical forms)
    # These often look like "interaction of X with Y" or "administered of X"
    if "interaction of" in chunk_text or \
       "inhibition of" in chunk_text or \
       "administered of" in chunk_text or \
       "treatment of" in chunk_text:
        is_prop = True

    head_match = re.search(r"\[(.*?)\]", chunk_text)
    tail_match = re.search(r"\{(.*?)\}", chunk_text)
    
    if head_match and tail_match:
        head = head_match.group(1).strip()
        tail = tail_match.group(1).strip()
        clean_text = chunk_text.replace("[", "").replace("]", "").replace("{", "").replace("}", "").strip()
        response = f"{head}|{relation}|{tail}"
        
        # If it has explicit markers, it's also a high-quality chunk (Smart or Proposition)
        return clean_text, response, True
    
    return chunk_text.strip(), relation, is_prop

def redefine_instruction(topn_path, knn_path, out_path):
    print(f"Processing {topn_path} -> {out_path} (Smart Permuted)...")
    count = 0
    
    with open(out_path, "w") as fw, open(knn_path, "r") as knn_fr:
        for idx, (topn_line, knn_raw) in enumerate(zip(_read_topn_lines(topn_path), knn_fr)):
            knn_obj = json.loads(knn_raw.strip())
            topn_sim_chuck = topn_line.get("topn_sim_chuck", {})

            # 1. Collect valid examples with Quality Score
            candidates = []
            for key, value in topn_sim_chuck.items():
                clean_ctx, full_response, is_prop = parse_chunk_triplet(key, value)
                if clean_ctx and full_response:
                    candidates.append({
                        "text": f"Context: {clean_ctx} Response: {full_response}",
                        "is_prop": is_prop
                    })
            
            # 2. Smart Selection
            # Sort candidates: Proposition chunks first!
            # Python's sort is stable, so original order is preserved among items with same 'is_prop'
            candidates.sort(key=lambda x: x["is_prop"], reverse=True)
            
            # Select Top 3 (now ensuring Proposition chunks are included if they exist)
            selected = [c["text"] for c in candidates[:3]]
            
            # 3. Generate Permutations
            permuted_examples = []
            if selected:
                # Generate all permutations of these 3 selected chunks
                for p in permutations(selected):
                    permuted_examples.append(" ".join([f"{i+1}. {ex}" for i, ex in enumerate(p)]))
            
            if not permuted_examples:
                permuted_examples = [""]

            # 4. Construct instruction (using first permutation)
            base_instruction = knn_obj.get("instruction", "")
            if "Examples:" in base_instruction:
                base_instruction = base_instruction.split("Examples:")[0].strip()
            
            examples_str = permuted_examples[0]
            
            knn_obj["instruction"] = f"{base_instruction} Examples: {examples_str}"
            
            fw.write(json.dumps(knn_obj))
            fw.write("\n")
            count += 1
            
    print(f"Done. Processed {count} lines.")

if __name__ == "__main__":
    import os
    if os.path.exists(HYBRID_TEST_INPUT):
        redefine_instruction(HYBRID_TEST_INPUT, KNN_TEST, OUTPUT_TEST)
    if os.path.exists(HYBRID_TRAIN_INPUT):
        redefine_instruction(HYBRID_TRAIN_INPUT, KNN_TRAIN, OUTPUT_TRAIN)


