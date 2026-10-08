import json
import re

def _read_topn_lines(topn_path):
    with open(topn_path, "r") as fr:
        for raw in fr:
            yield json.loads(raw)

def parse_chunk_triplet(chunk_text, relation):
    """
    Tries to extract [Head] and {Tail} from chunk text.
    Returns (clean_context, formatted_response) if successful, else (None, None).
    """
    head_match = re.search(r"\[(.*?)\]", chunk_text)
    tail_match = re.search(r"\{(.*?)\}", chunk_text)
    
    if head_match and tail_match:
        head = head_match.group(1).strip()
        tail = tail_match.group(1).strip()
        
        clean_text = chunk_text.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
        clean_text = clean_text.strip()
        
        response = f"{head}|{relation}|{tail}"
        return clean_text, response
    
    return None, None

def redefine_instruction(topn_path, knn_path, out_path):
    print(f"Processing {topn_path} -> {out_path}...")
    count = 0
    fallback_chunks = 0
    valid_chunks = 0
    
    with open(out_path, "w") as fw, open(knn_path, "r") as knn_fr:
        for idx, (topn_line, knn_raw) in enumerate(zip(_read_topn_lines(topn_path), knn_fr)):
            knn_obj = json.loads(knn_raw.strip())
            topn_sim_chuck = topn_line.get("topn_sim_chuck", {})

            # 1. Collect valid examples
            examples = []
            for key, value in topn_sim_chuck.items():
                key = key.strip()
                value = value.strip()
                
                if not key or not value or key == "{}" or value == "{}":
                    continue
                    
                # Parse triplet from chunk markers
                clean_ctx, full_response = parse_chunk_triplet(key, value)
                
                if clean_ctx and full_response:
                    examples.append(f"Context: {clean_ctx} Response: {full_response}")
                    valid_chunks += 1
                else:
                    # Fallback: Use original text and relation label if markers are missing
                    # This ensures we don't lose 90% of our examples
                    examples.append(f"Context: {key} Response: {value}")
                    fallback_chunks += 1

            # 2. Format examples cleanly
            formatted_examples = []
            for i, ex in enumerate(examples, 1):
                 formatted_examples.append(f"{i}. {ex}")
            
            examples_str = " ".join(formatted_examples)
            
            # 3. Construct instruction
            base_instruction = knn_obj.get("instruction", "")
            if "Examples:" in base_instruction:
                base_instruction = base_instruction.split("Examples:")[0].strip()
            
            if examples_str:
                knn_obj["instruction"] = f"{base_instruction} Examples: {examples_str}"
            else:
                knn_obj["instruction"] = f"{base_instruction} Examples: "

            fw.write(json.dumps(knn_obj))
            fw.write("\n")
            count += 1
            
    print(f"Done. Processed {count} lines.")
    print(f"Valid chunks used (Head|Rel|Tail): {valid_chunks}")
    print(f"Fallback chunks used (Rel only): {fallback_chunks}")

if __name__=="__main__":
    # Absolute KNN paths
    knn_train="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_train.json"
    knn_test="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_test.json"

    # Generate for test
    redefine_instruction(
        "test_data_topn_chucks_5_improved_deduplicated.json",
        knn_test,
        "test_chuck_instruction_5_llama13b_right_improved_deduplicated.json",
    )
    # Generate for train
    redefine_instruction(
        "train_data_topn_chucks_5_improved.json",
        knn_train,
        "train_chuck_instruction_5_llama13b_right_improved.json",
    )
