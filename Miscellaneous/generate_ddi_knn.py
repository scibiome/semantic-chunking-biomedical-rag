#!/usr/bin/env python3
"""
Generate DDI KNN files with correct instruction template.
Uses filtered DDI data (no NONE cases) for 4-way classification.

Follows the same structure as regenerate_chemprot_knn.py
"""
import json

# DDI instruction template (4 relations: MECHANISM, EFFECT, ADVISE, INT)
DDI_INSTRUCTION = (
    "please extract the triplet from this sentence, the triplet is [head entity, relation, tail entity], "
    "the element relation denotes the relationship between head entity and tail entity,  I will provide you "
    "the definition of the triplet you need to extract, the sentence from where your extract the triplets "
    "(head entity, relation, tail_entity) and the output format with examples. the relation must in my "
    "predefined relation set: ('MECHANISM', 'EFFECT', 'ADVISE', 'INT').  "
    "response Format: head entity|relation|tail entity. "
    "Examples: "
    "1. Context:The metabolism of Ketorolac can be decreased when combined with Fluconazole. "
    "Response: Ketorolac|MECHANISM|Fluconazole. "
    "2. Context:The risk or severity of adverse effects can be increased when Aliskiren is combined with Furosemide. "
    "Response: Aliskiren|EFFECT|Furosemide"
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
    print("Generating DDI KNN Files (Filtered Data - No NONE)", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    
    # Train (filtered - no NONE cases)
    print("📊 Generating TRAIN KNN file...", flush=True)
    train_gold = "../dataset/1_DDI/train_gmciht_format_no_none.jsonl"
    train_out = "KNN_ddi_train.json"
    n_train = build_knn(train_gold, train_out, DDI_INSTRUCTION)
    
    print(flush=True)
    
    # Test (filtered - no NONE cases)
    print("📊 Generating TEST KNN file...", flush=True)
    test_gold = "../dataset/1_DDI/test_gmciht_format_no_none.jsonl"
    test_out = "KNN_ddi_test.json"
    n_test = build_knn(test_gold, test_out, DDI_INSTRUCTION)
    
    print(flush=True)
    print("=" * 80, flush=True)
    print("✅ DDI KNN Files Generated Successfully!", flush=True)
    print("=" * 80, flush=True)
    print(f"  Train: {n_train} samples → KNN_ddi_train.json", flush=True)
    print(f"  Test:  {n_test} samples → KNN_ddi_test.json", flush=True)
    print(flush=True)
    print("✓ Correct template with 4 DDI relations:", flush=True)
    print("  ('MECHANISM', 'EFFECT', 'ADVISE', 'INT')", flush=True)
    print(flush=True)
    print("Note: Using filtered datasets (no NONE cases) matching test distribution", flush=True)
