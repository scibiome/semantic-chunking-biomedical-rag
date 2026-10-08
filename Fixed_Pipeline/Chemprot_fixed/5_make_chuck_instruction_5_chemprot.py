
import json
from itertools import permutations

def iter_permutation_strings(items, max_total=64):
    """
    Generate permutation strings following original logic:
    - include all permutations of length 1..n-1
    - include only the first permutation of length n
    Skip empty-length permutations.
    Yields strings like "Context: ... Response: ..." joined with spaces.
    """
    n = len(items)
    emitted = 0
    for i in range(1, max(n, 1)):
        for p in permutations(items, i):
            yield " ".join(p)
            emitted += 1
            if emitted >= max_total - 1:  # reserve one for the full-length perm
                break
        if emitted >= max_total - 1:
            break
    if n > 0 and emitted < max_total:
        full_iter = permutations(items, n)
        try:
            first_full = next(full_iter)
            yield " ".join(first_full)
        except StopIteration:
            pass

def _read_topn_lines(topn_path):
    with open(topn_path, "r") as fr:
        for raw in fr:
            yield json.loads(raw)

def redefine_instruction(topn_path, knn_path, out_path):
    with open(out_path, "w") as fw, open(knn_path, "r") as knn_fr:
        for idx, (topn_line, knn_raw) in enumerate(zip(_read_topn_lines(topn_path), knn_fr)):
            knn_obj = json.loads(knn_raw.strip())
            topn_sim_chuck = topn_line.get("topn_sim_chuck", {})

            items = []
            for key, value in topn_sim_chuck.items():
                items.append("Context: " + key + " Response: " + value)

            instruction = knn_obj.get("instruction", "")
            # Build in small chunks to avoid huge intermediate lists
            chunk = []
            count_in_chunk = 0
            first_added = False
            for s in iter_permutation_strings(items, max_total=64):
                if not s:
                    continue
                chunk.append(s)
                count_in_chunk += 1
                if count_in_chunk >= 32:
                    instruction = (instruction + ("|" if not first_added else "")) + "|".join(chunk)
                    first_added = True
                    chunk = []
                    count_in_chunk = 0
            if chunk:
                instruction = (instruction + ("|" if not first_added else "")) + "|".join(chunk)
            knn_obj["instruction"] = instruction

            fw.write(json.dumps(knn_obj))
            fw.write("\n")

if __name__=="__main__":
    print("=" * 70)
    print("Step 5: Creating instruction files for ChemProt")
    print("=" * 70)
    
    # ChemProt KNN paths (freshly generated from corrected triples datasets)
    knn_train="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_chemprot_train.json"
    knn_test="/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_chemprot_test.json"

    # Generate for test
    print("\n📊 Processing TEST data...")
    redefine_instruction(
        "test_data_topn_chucks_5_chemprot.json",
        knn_test,
        "test_chuck_instruction_5_llama13b_right_chemprot.json",
    )
    print("✅ Test instruction file created")
    
    # Generate for train
    print("\n📊 Processing TRAIN data...")
    redefine_instruction(
        "train_data_topn_chucks_5_chemprot.json",
        knn_train,
        "train_chuck_instruction_5_llama13b_right_chemprot.json",
    )
    print("✅ Train instruction file created")
    
    print("\n✅ Step 5 completed successfully!")
