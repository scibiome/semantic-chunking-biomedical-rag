import json
from pathlib import Path


BASE_DIR = Path("/home/users/riah/project_thesis/BIoMedRAG")

# Raw ADE files
TRAIN_RAW = BASE_DIR / "dataset" / "ade_classification" / "ade_corpus_v2_train_4000.json"
TEST_RAW = BASE_DIR / "dataset" / "ade_classification" / "ade_corpus_v2_test.json"

# Output instruction-style files
OUT_DIR = BASE_DIR / "2_relation_data_to_triple_train_data" / "ade_classification"
OUT_TRAIN = OUT_DIR / "train_ade_classification_instructions.json"
OUT_TEST = OUT_DIR / "test_ade_classification_instructions.json"


INSTRUCTION_TEXT = (
    "You are an excellent medical expert. The task is to predict the label for this sentence. "
    "The label is True if the sentence is ADE-related (describes an adverse drug event), "
    "and False otherwise. Respond with exactly one word: True or False."
)


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_split(raw_path: Path, out_path: Path):
    """
    Convert ADE JSONL (one object per line with fields `text` and `label`)
    into instruction / context / response / category format.
    """
    count = 0
    with raw_path.open("r", encoding="utf-8") as fr, out_path.open(
        "w", encoding="utf-8"
    ) as fw:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = obj.get("text", "").strip()
            label = obj.get("label")

            # Map 0/1 to False/True strings (as in the paper's ADE example)
            if label == 1 or label == "1":
                response = "True"
            else:
                response = "False"

            out_obj = {
                "instruction": INSTRUCTION_TEXT,
                "context": text,
                "response": response,
                "category": "classification",
            }
            fw.write(json.dumps(out_obj, ensure_ascii=False))
            fw.write("\n")
            count += 1

    print(f"Converted {count} examples from {raw_path} -> {out_path}")


if __name__ == "__main__":
    ensure_out_dir()
    convert_split(TRAIN_RAW, OUT_TRAIN)
    convert_split(TEST_RAW, OUT_TEST)


