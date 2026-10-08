import json
import random
import argparse
import sys

"""
S-HRT augmentation for **improved triple training files**, mirroring
`8_make_s_hrt_for_chunk_scorer_hybrid.py` but adapted to the improved files.

Goal:
  For each example, the `instruction` currently has:
    Examples: 1. Context: <chunk> Response: REL ...
  We want to rewrite these examples so they become full triplets:
    Context: HEAD <chunk> TAIL
    Response: HEAD|REL|TAIL

  HEAD/TAIL are sampled from a relation → [head|tail] mapping
  built on the dev file, exactly like the original S-HRT script.

Important:
  - We DO NOT touch the gold `response` field (still the true head|rel|tail).
  - We only change the in-prompt examples inside `instruction`.
"""

# Default paths (can be overridden via command-line arguments)
DEFAULT_TRAIN_PATH = "train_triplet_chunk_scorer_improved.json"
DEFAULT_TEST_PATH = "test_triplet_chunk_scorer_improved.json"
DEFAULT_DEV_SOURCE = "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"

# Presets for apply chunk scorer outputs (inputs = output files from apply_chunk_scorer.py)
VARIANT_PATHS = {
    "improved": ("train_triplet_chunk_scorer_improved.json", "test_triplet_chunk_scorer_improved.json", ""),
    "base_ner": ("train_triplet_chunk_scorer_base_ner.json", "test_triplet_chunk_scorer_base_ner.json", ""),
    "ddi_improved": ("train_triplet_chunk_scorer_ddi_improved.json", "test_triplet_chunk_scorer_ddi_improved.json", ""),
    "chemprot_improved": ("train_triplet_chunk_scorer_chemprot_improved.json", "test_triplet_chunk_scorer_chemprot_improved.json", ""),
    "chemprot_semantic_prop": ("train_triplet_chunk_scorer_chemprot_semantic_prop.json", "test_triplet_chunk_scorer_chemprot_semantic_prop.json", ""),
    "ddi_semantic_prop": ("train_triplet_chunk_scorer_ddi_semantic_prop.json", "test_triplet_chunk_scorer_ddi_semantic_prop.json", ""),
}
# Optional: dev file for building relation→(head,tail) mapping per variant (default: DEFAULT_DEV_SOURCE)
# ChemProt uses ACTIVATOR, DOWNREGULATOR, etc.; DDI uses MECHANISM, EFFECT, ADVISE, INT; GM-CIHT uses different relations.
VARIANT_DEV_SOURCE = {
    "ddi_improved": "/home/users/riah/project_thesis/BIoMedRAG/dataset/1_DDI/DDI_Dataset/dev_gmciht_format_no_none.jsonl",
    "ddi_semantic_prop": "/home/users/riah/project_thesis/BIoMedRAG/dataset/1_DDI/DDI_Dataset/dev_gmciht_format_no_none.jsonl",
    "chemprot_improved": "/home/users/riah/project_thesis/BIoMedRAG/dataset/2_chemprot/dev_triples_gmciht_format_SINGLE.jsonl",
    "chemprot_semantic_prop": "/home/users/riah/project_thesis/BIoMedRAG/dataset/2_chemprot/dev_triples_gmciht_format_SINGLE.jsonl",
}


def build_relation_ht_mapping(dev_source_path: str):
    dic_ = {}
    with open(dev_source_path) as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            h = obj["SUBJECT_TEXT"]
            t = obj["OBJECT_TEXT"]
            r = obj["PREDICATE"]
            dic_.setdefault(r, []).append(h + "|" + t)
    return dic_


def parse_examples(examples_str: str):
    """
    Parse the 'Examples:' tail into a list of (chunk_text, relation) pairs.
    Expected input pattern (like current improved files):
      "1. Context: chunk1 Response: REL1 2. Context: chunk2 Response: REL2 ..."
    We parse up to all examples we can find.
    """
    ex = examples_str.strip()
    if not ex:
        return []

    results = []
    # Split on 'Context:' tokens
    parts = ex.split("Context:")
    # parts[0] is before first Context:, skip
    for seg in parts[1:]:
        seg = seg.strip()
        if not seg:
            continue
        # seg looks like: "<chunk text> Response: REL [2. Context: ...]"
        if "Response:" not in seg:
            continue
        chunk_part, resp_part = seg.split("Response:", 1)
        chunk_text = chunk_part.strip()
        # relation token is first token before next "Context:" or end
        # e.g. "TREATS 2. Context: ..." -> "TREATS"
        rel_block = resp_part.split("Context:")[0]
        rel_token = rel_block.strip().split()[0] if rel_block.strip() else ""
        rel_token = rel_token.strip().strip(".,")
        if not chunk_text or not rel_token:
            continue
        results.append((chunk_text, rel_token))
    return results


def make_s_hrt_examples(examples_str: str, r_ht_mapping: dict, max_examples: int = 1) -> str:
    """
    Turn relation-only examples into S-HRT style examples using R_HT_mapping.
    For each (chunk, REL) we build:
      Context: HEAD + chunk + TAIL
      Response: HEAD|REL|TAIL

    We follow the original script behavior: typically use only the first example,
    but we can extend to more if needed.
    """
    pairs = parse_examples(examples_str)
    if not pairs:
        return ""

    renew_exc = []
    # Use up to max_examples per line
    for idx, (chunk_text, rel) in enumerate(pairs[:max_examples], start=1):
        hts = r_ht_mapping.get(rel)
        if not hts:
            continue
        head_tail = random.sample(hts, 1)[0].split("|")
        if len(head_tail) != 2:
            continue
        head, tail = head_tail
        renew_input = head + " " + chunk_text + " " + tail
        renew_response = f"{head}|{rel}|{tail}"
        new_e = f"{idx}. Context: {renew_input} Response: {renew_response}"
        renew_exc.append(new_e)

    return " ".join(renew_exc)


def process_file(input_path: str, output_path: str, is_test: bool, max_examples: int, r_ht_mapping: dict) -> None:
    i = -1
    updated = 0
    with open(input_path) as fr, open(output_path, "w") as fw:
        for line in fr:
            i += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"ERROR at line {i+1}: {e}")
                print(f"  First 200 chars: {raw[:200]}")
                continue
            instr = obj.get("instruction", "")
            if "Examples:" not in instr:
                fw.write(json.dumps(obj))
                fw.write("\n")
                continue

            base_instr, examples = instr.split("Examples:", 1)
            base_instr = base_instr.strip()
            examples = examples.strip()

            s_hrt_examples = make_s_hrt_examples(examples, r_ht_mapping, max_examples=max_examples)
            if s_hrt_examples:
                updated += 1
                new_instruction = f"{base_instr} Examples: {s_hrt_examples}".strip()
                obj["instruction"] = new_instruction
            else:
                # If we can't build S-HRT examples, keep original instruction
                obj["instruction"] = instr.strip()

            fw.write(json.dumps(obj))
            fw.write("\n")
            fw.flush()

    split = "TEST" if is_test else "TRAIN"
    print(f"{split}: updated {updated} instructions with S-HRT examples (out of {i+1} lines).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create S-HRT format from improved chunk scorer outputs")
    parser.add_argument("--variant", type=str, choices=list(VARIANT_PATHS), default=None,
                        help="Preset: 'improved', 'base_ner', 'ddi_improved', 'chemprot_improved', 'chemprot_semantic_prop', or 'ddi_semantic_prop'. Overrides --train-input/--test-input/--output-suffix.")
    parser.add_argument("--train-input", type=str, default=DEFAULT_TRAIN_PATH,
                        help="Input train file path (default: train_triplet_chunk_scorer_improved.json)")
    parser.add_argument("--test-input", type=str, default=DEFAULT_TEST_PATH,
                        help="Input test file path (default: test_triplet_chunk_scorer_improved.json)")
    parser.add_argument("--dev-source", type=str, default=DEFAULT_DEV_SOURCE,
                        help="Dev file for building relation mapping (default: GM-CIHT dev file)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix to add to output filenames (e.g., '_improved')")
    parser.add_argument("--max-examples", type=int, default=1,
                        help="Maximum number of S-HRT examples per instruction (default: 1)")
    parser.add_argument("--output-suffix-3", type=str, default="",
                        help="Suffix for 3-example version (default: same as output-suffix)")
    
    args = parser.parse_args()
    
    if args.variant:
        train_path, test_path, out_suffix = VARIANT_PATHS[args.variant]
        args.train_input = train_path
        args.test_input = test_path
        args.output_suffix = out_suffix
        args.output_suffix_3 = out_suffix
        if args.variant in VARIANT_DEV_SOURCE:
            args.dev_source = VARIANT_DEV_SOURCE[args.variant]
            print(f"Using variant '{args.variant}': train={train_path}, test={test_path}, dev={args.dev_source}")
        else:
            print(f"Using variant '{args.variant}': train={train_path}, test={test_path}")
    
    # Build relation mapping
    print(f"Building relation mapping from: {args.dev_source}")
    r_ht_mapping = build_relation_ht_mapping(args.dev_source)
    print(f"Loaded {len(r_ht_mapping)} relations")
    
    # Determine output filenames
    train_base = args.train_input.replace(".json", "")
    test_base = args.test_input.replace(".json", "")
    
    suffix = args.output_suffix
    suffix_3 = args.output_suffix_3 if args.output_suffix_3 else suffix
    
    output_train_s_hrt = f"{train_base}_s_hrt{suffix}.json"
    output_test_s_hrt = f"{test_base}_s_hrt{suffix}.json"
    output_train_s_hrt3 = f"{train_base}_s_hrt3{suffix_3}.json"
    output_test_s_hrt3 = f"{test_base}_s_hrt3{suffix_3}.json"
    
    print(f"\nProcessing files:")
    print(f"  Train input: {args.train_input}")
    print(f"  Test input: {args.test_input}")
    print(f"  Output (1 example): {output_train_s_hrt}, {output_test_s_hrt}")
    print(f"  Output (3 examples): {output_train_s_hrt3}, {output_test_s_hrt3}")
    print()
    
    # Version 1: single S-HRT example per line
    process_file(
        args.test_input,
        output_test_s_hrt,
        is_test=True,
        max_examples=1,
        r_ht_mapping=r_ht_mapping,
    )
    process_file(
        args.train_input,
        output_train_s_hrt,
        is_test=False,
        max_examples=1,
        r_ht_mapping=r_ht_mapping,
    )

    # Version 2: up to 3 S-HRT examples per line
    process_file(
        args.test_input,
        output_test_s_hrt3,
        is_test=True,
        max_examples=3,
        r_ht_mapping=r_ht_mapping,
    )
    process_file(
        args.train_input,
        output_train_s_hrt3,
        is_test=False,
        max_examples=3,
        r_ht_mapping=r_ht_mapping,
    )
    
    print(f"\n✅ All S-HRT files generated successfully!")
    print(f"  Files created:")
    print(f"    - {output_train_s_hrt}")
    print(f"    - {output_test_s_hrt}")
    print(f"    - {output_train_s_hrt3}")
    print(f"    - {output_test_s_hrt3}")

