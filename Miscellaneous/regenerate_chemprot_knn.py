#!/usr/bin/env python3
"""
Regenerate ChemProt KNN files with correct instruction template.
"""
import json

# Correct ChemProt instruction template (5 relations only)
CHEMPROT_INSTRUCTION = (
    "please extract the triplet from this sentence, the triplet is [head entity, relation, tail entity], "
    "the element relation denotes the relationship between head entity and tail entity,  I will provide you "
    "the definition of the triplet you need to extract, the sentence from where your extract the triplets "
    "(head entity, relation, tail_entity) and the output format with examples. the relation must in my "
    "predefined relation set: ('ACTIVATOR', 'AGONIST', 'ANTAGONIST', 'DOWNREGULATOR', 'PRODUCT-OF').  "
    "response Format: head entity|relation|tail entity. "
    "Examples: "
    "1. Context:Tomudex treatment resulted in increase in cyclin E expression. Response: Tomudex|ACTIVATOR|cyclin E. "
    "2. Context:Dopamine acts as an agonist at D2 receptors. Response: Dopamine|AGONIST|D2 receptors"
)

def build_knn(gold_path, out_path, instr_template):
    """Build KNN file from GM-CIHT format triples"""
    count = 0
    print(f"  Reading: {gold_path}", flush=True)
    with open(gold_path, "r") as fr, open(out_path, "w") as fw:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            head = obj.get("SUBJECT_TEXT", "").strip()
            rel = obj.get("PREDICATE", "").strip()
            tail = obj.get("OBJECT_TEXT", "").strip()
            sent = obj.get("SENTENCE", "").strip()
            
            knn_entry = {
                "instruction": instr_template,
                "context": sent,  # Only sentence - triple extraction task
                "response": f"{head}|{rel}|{tail}",
                "category": "triplet_extraction",
            }
            fw.write(json.dumps(knn_entry))
            fw.write("\n")
            count += 1
            
            if count % 500 == 0:
                print(f"    Processed {count} samples...", flush=True)
    
    print(f"  ✓ Wrote {count} entries to {out_path}", flush=True)
    return count

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("Regenerating ChemProt KNN Files with Correct Template", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    
    # Train
    print("📊 Generating TRAIN KNN file...", flush=True)
    train_gold = "../dataset/2_chemprot/train_triples_gmciht_format.jsonl"
    train_out = "KNN_chemprot_train.json"
    n_train = build_knn(train_gold, train_out, CHEMPROT_INSTRUCTION)
    
    print(flush=True)
    
    # Test
    print("📊 Generating TEST KNN file...", flush=True)
    test_gold = "../dataset/2_chemprot/test_triples_gmciht_format.jsonl"
    test_out = "KNN_chemprot_test.json"
    n_test = build_knn(test_gold, test_out, CHEMPROT_INSTRUCTION)
    
    print(flush=True)
    print("=" * 80, flush=True)
    print("✅ KNN Files Regenerated Successfully!", flush=True)
    print("=" * 80, flush=True)
    print(f"  Train: {n_train} samples → KNN_chemprot_train.json", flush=True)
    print(f"  Test:  {n_test} samples → KNN_chemprot_test.json", flush=True)
    print(flush=True)
    print("✓ Correct template with 5 ChemProt relations:", flush=True)
    print("  ('ACTIVATOR', 'AGONIST', 'ANTAGONIST', 'DOWNREGULATOR', 'PRODUCT-OF')", flush=True)

