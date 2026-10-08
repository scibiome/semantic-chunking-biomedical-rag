#!/usr/bin/env python3
"""
Build KNN training data for ChemProt (Triple Extraction)
Similar to GM-CIHT: context = sentence only, response = subject|relation|object
"""
import json
import argparse
from pathlib import Path


def load_instruction_template():
    """
    ChemProt instruction template for triple extraction.
    Similar to GM-CIHT but with ChemProt-specific relations (5 types only).
    """
    return (
        "please extract the triplet from this sentence, the triplet is [head entity, relation, tail entity], "
        "the element relation denotes the relationship between head entity and tail entity. I will provide you "
        "the definition of the triplet you need to extract, the sentence from where you extract the triplets "
        "(head entity, relation, tail_entity) and the output format with examples. the relation must be in my "
        "predefined relation set: ('ACTIVATOR', 'AGONIST', 'ANTAGONIST', 'DOWNREGULATOR', 'PRODUCT-OF'). "
        "response Format: head entity|relation|tail entity. "
        "Examples: "
        "1. Context: Tomudex treatment resulted in increase in cyclin E expression. "
        "Response: Tomudex|ACTIVATOR|cyclin E. "
        "2. Context: Dopamine acts as an agonist at D2 receptors. "
        "Response: Dopamine|AGONIST|D2 receptors"
    )


def build_knn_from_chunks(chunks_file: Path, out_path: Path, instr_template: str):
    """
    Build KNN data from chunked sentences.
    
    Args:
        chunks_file: JSONL file with chunked sentences
        out_path: Output path for KNN data
        instr_template: Instruction template string
    """
    count = 0
    with open(chunks_file, "r") as fr, open(out_path, "w") as fw:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            
            # Extract fields
            sentence = obj.get("sentence", "").strip()
            subject = obj.get("subject", "").strip()
            relation = obj.get("relation", "").strip()
            object_entity = obj.get("object", "").strip()
            
            # Create KNN entry - context is just the sentence (triple extraction)
            knn_entry = {
                "instruction": instr_template,
                "context": sentence,  # Only sentence, NO entity markers
                "response": f"{subject}|{relation}|{object_entity}",
                "category": "triplet_extraction",
            }
            
            fw.write(json.dumps(knn_entry))
            fw.write("\n")
            count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build KNN training data for ChemProt triple extraction"
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to chunked JSONL file (e.g., chemprot_train_chunks_5.jsonl)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output KNN path (e.g., chemprot_KNN_demo_train.json)"
    )
    
    args = parser.parse_args()
    
    template = load_instruction_template()
    n = build_knn_from_chunks(Path(args.chunks), Path(args.out), template)
    print(f"✅ Wrote {n} entries to {args.out}")


if __name__ == "__main__":
    main()


