import json
import re
from itertools import permutations

# Define file paths
KNN_TRAIN = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_train.json"
KNN_TEST = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_test.json"

HYBRID_TRAIN_INPUT = "hybrid_train_data_topn_scored_5.json"
HYBRID_TEST_INPUT = "hybrid_test_data_topn_scored_5.json"

OUTPUT_TRAIN = "hybrid_train_instruction_5_scored.json"
OUTPUT_TEST = "hybrid_test_instruction_5_scored.json"

def _read_topn_lines(topn_path):
    with open(topn_path, "r") as fr:
        for raw in fr:
            yield json.loads(raw)

def parse_chunk_triplet(chunk_text, relation):
    """
    Extracts [Head] and {Tail} if present, or uses robust fallback.
    Returns: (clean_text, response)
    """
    head_match = re.search(r"\[(.*?)\]", chunk_text)
    tail_match = re.search(r"\{(.*?)\}", chunk_text)
    
    if head_match and tail_match:
        head = head_match.group(1).strip()
        tail = tail_match.group(1).strip()
        # Remove markers for the Context
        clean_text = chunk_text.replace("[", "").replace("]", "").replace("{", "").replace("}", "").strip()
        # Keep markers in mind but response needs strict format?
        # Standard format: Response: Head|Relation|Tail
        response = f"{head}|{relation}|{tail}"
        return clean_text, response
    
    # Fallback for Fixed chunks (no markers)
    # We don't know the exact head/tail in the chunk, so we use the 
    # original sentence's head/tail if available, or just the relation?
    # Actually, previous scripts just used the Relation or assumed the chunk implied them.
    # But for Training Data, we need the Reference Response.
    # The 'relation' passed here is just the Relation Label.
    
    # Wait, the instruction examples need a Response.
    # In 'fixed' pipeline, chunks didn't have markers. How did they get Head/Tail?
    # They usually didn't! They often just had "Response: Relation".
    # BUT, the Triple Extraction Model is trained to output triplets.
    # If the examples only show Relation, the model might learn to only output Relation.
    
    # Let's check the Fixed Chunking output format from previous logs.
    # "Context: ... Response: PROCESS_OF" (Relation only?)
    # If so, we should stick to that for Consistency if we want Relation Extraction.
    # BUT, our goal is Triplet Extraction.
    
    # If we have Proposition Chunks (with Head/Tail), we SHOULD use them!
    # If we have Fixed Chunks (no Head/Tail), we can only provide Relation.
    
    return chunk_text.strip(), relation

def redefine_instruction(topn_path, knn_path, out_path):
    print(f"Processing {topn_path} -> {out_path} (Scored + Permuted)...")
    count = 0
    
    with open(out_path, "w") as fw, open(knn_path, "r") as knn_fr:
        for idx, (topn_line, knn_raw) in enumerate(zip(_read_topn_lines(topn_path), knn_fr)):
            knn_obj = json.loads(knn_raw.strip())
            topn_sim_chuck = topn_line.get("topn_sim_chuck", {})

            # 1. Collect Valid Examples
            # The dictionary is ALREADY sorted by Semantic Score from Step 3.
            # We just take the top ones.
            candidates = []
            for key, value in topn_sim_chuck.items():
                clean_ctx, full_response = parse_chunk_triplet(key, value)
                if clean_ctx and full_response:
                    candidates.append(f"Context: {clean_ctx} Response: {full_response}")
            
            # 2. Select Top 3
            selected = candidates[:3]
            
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






import re
from itertools import permutations

# Define file paths
KNN_TRAIN = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_train.json"
KNN_TEST = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_test.json"

HYBRID_TRAIN_INPUT = "hybrid_train_data_topn_scored_5.json"
HYBRID_TEST_INPUT = "hybrid_test_data_topn_scored_5.json"

OUTPUT_TRAIN = "hybrid_train_instruction_5_scored.json"
OUTPUT_TEST = "hybrid_test_instruction_5_scored.json"

def _read_topn_lines(topn_path):
    with open(topn_path, "r") as fr:
        for raw in fr:
            yield json.loads(raw)

def parse_chunk_triplet(chunk_text, relation):
    """
    Extracts [Head] and {Tail} if present, or uses robust fallback.
    Returns: (clean_text, response)
    """
    head_match = re.search(r"\[(.*?)\]", chunk_text)
    tail_match = re.search(r"\{(.*?)\}", chunk_text)
    
    if head_match and tail_match:
        head = head_match.group(1).strip()
        tail = tail_match.group(1).strip()
        # Remove markers for the Context
        clean_text = chunk_text.replace("[", "").replace("]", "").replace("{", "").replace("}", "").strip()
        # Keep markers in mind but response needs strict format?
        # Standard format: Response: Head|Relation|Tail
        response = f"{head}|{relation}|{tail}"
        return clean_text, response
    
    # Fallback for Fixed chunks (no markers)
    # We don't know the exact head/tail in the chunk, so we use the 
    # original sentence's head/tail if available, or just the relation?
    # Actually, previous scripts just used the Relation or assumed the chunk implied them.
    # But for Training Data, we need the Reference Response.
    # The 'relation' passed here is just the Relation Label.
    
    # Wait, the instruction examples need a Response.
    # In 'fixed' pipeline, chunks didn't have markers. How did they get Head/Tail?
    # They usually didn't! They often just had "Response: Relation".
    # BUT, the Triple Extraction Model is trained to output triplets.
    # If the examples only show Relation, the model might learn to only output Relation.
    
    # Let's check the Fixed Chunking output format from previous logs.
    # "Context: ... Response: PROCESS_OF" (Relation only?)
    # If so, we should stick to that for Consistency if we want Relation Extraction.
    # BUT, our goal is Triplet Extraction.
    
    # If we have Proposition Chunks (with Head/Tail), we SHOULD use them!
    # If we have Fixed Chunks (no Head/Tail), we can only provide Relation.
    
    return chunk_text.strip(), relation

def redefine_instruction(topn_path, knn_path, out_path):
    print(f"Processing {topn_path} -> {out_path} (Scored + Permuted)...")
    count = 0
    
    with open(out_path, "w") as fw, open(knn_path, "r") as knn_fr:
        for idx, (topn_line, knn_raw) in enumerate(zip(_read_topn_lines(topn_path), knn_fr)):
            knn_obj = json.loads(knn_raw.strip())
            topn_sim_chuck = topn_line.get("topn_sim_chuck", {})

            # 1. Collect Valid Examples
            # The dictionary is ALREADY sorted by Semantic Score from Step 3.
            # We just take the top ones.
            candidates = []
            for key, value in topn_sim_chuck.items():
                clean_ctx, full_response = parse_chunk_triplet(key, value)
                if clean_ctx and full_response:
                    candidates.append(f"Context: {clean_ctx} Response: {full_response}")
            
            # 2. Select Top 3
            selected = candidates[:3]
            
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











