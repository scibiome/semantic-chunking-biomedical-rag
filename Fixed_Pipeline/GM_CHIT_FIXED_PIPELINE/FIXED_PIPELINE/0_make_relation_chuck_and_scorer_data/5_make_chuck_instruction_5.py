import json

def _read_topn_lines(topn_path):
    with open(topn_path, "r") as fr:
        for raw in fr:
            yield json.loads(raw)

def redefine_instruction(topn_path, knn_path, out_path):
    print(f"Processing {topn_path} -> {out_path}...")
    count = 0
    with open(out_path, "w") as fw, open(knn_path, "r") as knn_fr:
        for idx, (topn_line, knn_raw) in enumerate(zip(_read_topn_lines(topn_path), knn_fr)):
            knn_obj = json.loads(knn_raw.strip())
            topn_sim_chuck = topn_line.get("topn_sim_chuck", {})

            # 1. Collect valid examples
            examples = []
            for key, value in topn_sim_chuck.items():
                key = key.strip()
                value = value.strip()
                # Filter out empty or placeholder artifacts
                if not key or not value or key == "{}" or value == "{}":
                    continue
                
                # REVERTED LOGIC: Just use key/value as is (Context: ... Response: RELATION)
                # This matches the logic used for the Fixed Model that got 74%
                examples.append(f"Context: {key} Response: {value}")

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

if __name__=="__main__":
    # Absolute KNN paths
    knn_train="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_train.json"
    knn_test="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_test.json"

    # Standard Fixed pipeline paths
    redefine_instruction(
        "test_data_topn_chucks_5.json", 
        knn_test,
        "test_chuck_instruction_5_llama13b_right.json",
    )
    redefine_instruction(
        "train_data_topn_chucks_5.json",
        knn_train,
        "train_chuck_instruction_5_llama13b_right.json",
    )
