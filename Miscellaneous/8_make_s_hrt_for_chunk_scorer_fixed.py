import json
import random

"""
S-HRT augmentation for **fixed triple training files**, mirroring
the hybrid S-HRT script but adapted to the fixed pipeline files.

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

# Fixed triple files (after apply_chunk_scorer)
train_fixed_path = "train_triplet_chunk_scorer.json"
test_fixed_path = "test_triplet_chunk_scorer.json"

# Dev file used to build relation -> [head|tail] mapping
dev_source = "/home/users/riah/project_thesis/BIoMedRAG/dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl"


def build_relation_ht_mapping():
    dic_ = {}
    with open(dev_source) as fr:
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


R_HT_mapping = build_relation_ht_mapping()


def parse_examples(examples_str: str):
    """
    Parse the 'Examples:' tail into a list of (chunk_text, relation) pairs.
    Expected input pattern (like current fixed files):
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


def make_s_hrt_examples(examples_str: str, max_examples: int = 1) -> str:
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
        hts = R_HT_mapping.get(rel)
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


def process_file(input_path: str, output_path: str, is_test: bool, max_examples: int) -> None:
    i = -1
    updated = 0
    with open(input_path) as fr, open(output_path, "w") as fw:
        for line in fr:
            i += 1
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            instr = obj.get("instruction", "")
            if "Examples:" not in instr:
                fw.write(json.dumps(obj))
                fw.write("\n")
                continue

            base_instr, examples = instr.split("Examples:", 1)
            base_instr = base_instr.strip()
            examples = examples.strip()

            s_hrt_examples = make_s_hrt_examples(examples, max_examples=max_examples)
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
    # Version 1: single S-HRT example per line
    print("Processing fixed pipeline files with 1 S-HRT example...")
    process_file(
        test_fixed_path,
        "test_triplet_chunk_scorer_fixed_s_hrt.json",
        is_test=True,
        max_examples=1,
    )
    process_file(
        train_fixed_path,
        "train_triplet_chunk_scorer_fixed_s_hrt.json",
        is_test=False,
        max_examples=1,
    )

    # Version 2: up to 3 S-HRT examples per line
    print("\nProcessing fixed pipeline files with 3 S-HRT examples...")
    process_file(
        test_fixed_path,
        "test_triplet_chunk_scorer_fixed_s_hrt3.json",
        is_test=True,
        max_examples=3,
    )
    process_file(
        train_fixed_path,
        "train_triplet_chunk_scorer_fixed_s_hrt3.json",
        is_test=False,
        max_examples=3,
    )

    print("\n✅ S-HRT augmentation complete for fixed pipeline files!")

