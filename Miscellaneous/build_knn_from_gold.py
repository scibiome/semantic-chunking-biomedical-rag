#!/usr/bin/env python3
"""
Rebuild KNN_demo_test.json (and optionally train) from the gold dataset.
- Reads the instruction template from an existing KNN file to preserve format
- Outputs one entry per gold line: {instruction, context, response, category}
"""
import json
import argparse
from pathlib import Path


def load_instruction_template(knn_paths):
    for p in knn_paths:
        try:
            with open(p, "r") as f:
                first = f.readline().strip()
                if first:
                    obj = json.loads(first)
                    instr = obj.get("instruction")
                    if instr:
                        return instr
        except Exception:
            continue
    # Fallback minimal template if none found
    return (
        "please extract the triplet from this sentence, the triplet is [head entity, relation, tail entity], "
        "the element relation denotes the relationship between head entity and tail entity,  I will provide you "
        "the definition of the triplet you need to extract, the sentence from where your extract the triplets "
        "(head entity, relation, tail_entity) and the output format with examples. the relation must in my "
        "predefined relation set: ('PRECEDES', 'PREDISPOSES', 'TREATS', 'AFFECTS', 'INTERACTS_WITH', 'INHIBITS', "
        "'PRODUCES', 'ADMINISTERED_TO', 'PROCESS_OF', 'AUGMENTS', 'PREVENTS', 'DIAGNOSES', 'COEXISTS_WITH', "
        "'ASSOCIATED_WITH', 'DISRUPTS', 'CAUSES', 'COMPLICATES', 'SYMPTOM_OF', 'DOES_NOT_TREAT', 'STIMULATES', "
        "'MANIFESTATION_OF', 'USES').  response Format: head entity|relation|tail entity. Examples: 1. Context:{} "
        "Response: {} 2. Context:{}. Response: {}"
    )


def build_knn(gold_path: Path, out_path: Path, instr_template: str):
    count = 0
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
                "context": sent,
                "response": f"{head}|{rel}|{tail}",
                "category": "triplet_extraction",
            }
            fw.write(json.dumps(knn_entry))
            fw.write("\n")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to gold JSONL (e.g., renew_triplet_T_test.jsonl)")
    ap.add_argument("--out", required=True, help="Output KNN path (e.g., KNN_demo_test.json)")
    ap.add_argument("--template-from", nargs="*", default=[], help="Existing KNN files to read instruction template from (first line)")
    args = ap.parse_args()

    template = load_instruction_template(args.template_from)
    n = build_knn(Path(args.gold), Path(args.out), template)
    print(f"✅ Wrote {n} entries to {args.out}")


if __name__ == "__main__":
    main()



