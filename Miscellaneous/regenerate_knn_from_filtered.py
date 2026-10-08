#!/usr/bin/env python3
"""
Regenerate KNN files from FILTERED datasets (SINGLE versions - one triplet per sentence).
This fixes the multi-triplet issue by ensuring each sentence appears only once.
"""
import json
from pathlib import Path

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

# ChemProt instruction template (5 relations only)
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

def build_knn(gold_path: Path, out_path: Path, instr_template: str, dataset_name: str):
    """Build KNN file from GM-CIHT format triples"""
    count = 0
    print(f"  Reading: {gold_path}", flush=True)
    
    if not gold_path.exists():
        print(f"  ⚠️  File not found: {gold_path}", flush=True)
        return 0
    
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
            
            if not sent or not head or not tail or not rel:
                continue
            
            knn_entry = {
                "instruction": instr_template,
                "context": sent,  # Only sentence - triple extraction task
                "response": f"{head}|{rel}|{tail}",
                "category": "triplet_extraction",
            }
            fw.write(json.dumps(knn_entry, ensure_ascii=False))
            fw.write("\n")
            count += 1
            
            if count % 500 == 0:
                print(f"    Processed {count} samples...", flush=True)
    
    print(f"  ✓ Wrote {count} entries to {out_path}", flush=True)
    return count

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("Regenerating KNN Files from FILTERED Datasets (SINGLE - One Triplet Per Sentence)", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    
    base_dir = Path(__file__).parent
    dataset_dir = base_dir.parent / "dataset"
    
    # DDI
    print("📊 DDI - Generating KNN files from FILTERED data...", flush=True)
    print("-" * 80, flush=True)
    
    # Check if train file exists (might need to create from original)
    ddi_train_filtered = dataset_dir / "1_DDI" / "train_gmciht_format_no_none_SINGLE.jsonl"
    ddi_test_filtered = dataset_dir / "1_DDI" / "test_gmciht_format_no_none_SINGLE.jsonl"
    ddi_dev_filtered = dataset_dir / "1_DDI" / "dev_gmciht_format_no_none_SINGLE.jsonl"
    
    # If train filtered doesn't exist, check if we need to create it
    if not ddi_train_filtered.exists():
        ddi_train_orig = dataset_dir / "1_DDI" / "train_gmciht_format_no_none.jsonl"
        if ddi_train_orig.exists():
            print(f"  Creating filtered train file from: {ddi_train_orig}", flush=True)
            seen = set()
            filtered = []
            with open(ddi_train_orig, 'r') as f_in:
                for line in f_in:
                    data = json.loads(line.strip())
                    sent = data.get('SENTENCE', '').strip()
                    if sent not in seen:
                        filtered.append(data)
                        seen.add(sent)
            with open(ddi_train_filtered, 'w') as f_out:
                for data in filtered:
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f"  ✓ Created: {ddi_train_filtered} ({len(filtered)} examples)", flush=True)
    
    print(flush=True)
    print("  📝 TRAIN KNN...", flush=True)
    ddi_train_knn = base_dir / "KNN_ddi_train_SINGLE.json"
    n_train = build_knn(ddi_train_filtered, ddi_train_knn, DDI_INSTRUCTION, "DDI")
    
    print(flush=True)
    print("  📝 TEST KNN...", flush=True)
    ddi_test_knn = base_dir / "KNN_ddi_test_SINGLE.json"
    n_test = build_knn(ddi_test_filtered, ddi_test_knn, DDI_INSTRUCTION, "DDI")
    
    print(flush=True)
    
    # ChemProt
    print("📊 ChemProt - Generating KNN files from FILTERED data...", flush=True)
    print("-" * 80, flush=True)
    
    chemprot_train_filtered = dataset_dir / "2_chemprot" / "train_triples_gmciht_format_SINGLE.jsonl"
    chemprot_test_filtered = dataset_dir / "2_chemprot" / "test_triples_gmciht_format_SINGLE.jsonl"
    chemprot_dev_filtered = dataset_dir / "2_chemprot" / "dev_triples_gmciht_format_SINGLE.jsonl"
    
    # If train filtered doesn't exist, create it
    if not chemprot_train_filtered.exists():
        chemprot_train_orig = dataset_dir / "2_chemprot" / "train_triples_gmciht_format.jsonl"
        if chemprot_train_orig.exists():
            print(f"  Creating filtered train file from: {chemprot_train_orig}", flush=True)
            seen = set()
            filtered = []
            with open(chemprot_train_orig, 'r') as f_in:
                for line in f_in:
                    data = json.loads(line.strip())
                    sent = data.get('SENTENCE', '').strip()
                    if sent not in seen:
                        filtered.append(data)
                        seen.add(sent)
            with open(chemprot_train_filtered, 'w') as f_out:
                for data in filtered:
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f"  ✓ Created: {chemprot_train_filtered} ({len(filtered)} examples)", flush=True)
    
    # If test filtered doesn't exist, create it
    if not chemprot_test_filtered.exists():
        chemprot_test_orig = dataset_dir / "2_chemprot" / "test_triples_gmciht_format.jsonl"
        if chemprot_test_orig.exists():
            print(f"  Creating filtered test file from: {chemprot_test_orig}", flush=True)
            seen = set()
            filtered = []
            with open(chemprot_test_orig, 'r') as f_in:
                for line in f_in:
                    data = json.loads(line.strip())
                    sent = data.get('SENTENCE', '').strip()
                    if sent not in seen:
                        filtered.append(data)
                        seen.add(sent)
            with open(chemprot_test_filtered, 'w') as f_out:
                for data in filtered:
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            print(f"  ✓ Created: {chemprot_test_filtered} ({len(filtered)} examples)", flush=True)
    
    print(flush=True)
    print("  📝 TRAIN KNN...", flush=True)
    chemprot_train_knn = base_dir / "KNN_chemprot_train_SINGLE.json"
    n_chemprot_train = build_knn(chemprot_train_filtered, chemprot_train_knn, CHEMPROT_INSTRUCTION, "ChemProt")
    
    print(flush=True)
    print("  📝 TEST KNN...", flush=True)
    chemprot_test_knn = base_dir / "KNN_chemprot_test_SINGLE.json"
    n_chemprot_test = build_knn(chemprot_test_filtered, chemprot_test_knn, CHEMPROT_INSTRUCTION, "ChemProt")
    
    print(flush=True)
    print("=" * 80, flush=True)
    print("✅ KNN Files Regenerated from FILTERED Datasets!", flush=True)
    print("=" * 80, flush=True)
    print(f"  DDI Train: {n_train} samples → KNN_ddi_train_SINGLE.json", flush=True)
    print(f"  DDI Test:  {n_test} samples → KNN_ddi_test_SINGLE.json", flush=True)
    print(f"  ChemProt Train: {n_chemprot_train} samples → KNN_chemprot_train_SINGLE.json", flush=True)
    print(f"  ChemProt Test:  {n_chemprot_test} samples → KNN_chemprot_test_SINGLE.json", flush=True)
    print(flush=True)
    print("✓ All files use SINGLE format (one triplet per sentence)", flush=True)
    print("✓ This fixes the multi-triplet training issue", flush=True)



