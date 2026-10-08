import json
from pathlib import Path

TRAIN_TOPN = Path("train_data_topn_chucks_ade_5.json")
TEST_TOPN = Path("test_data_topn_chucks_ade_5.json")
KNN_TRAIN  = Path("KNN_ade_train_classification.json")
KNN_TEST   = Path("KNN_ade_test_classification.json")

INSTRUCTION = (
    "You are an excellent medical expert. The task is to predict the label "
    "for this sentence. The label is True if the sentence is ADE-related "
    "and False otherwise."
)

def convert(topn_path, knn_path):
    with topn_path.open() as fr, knn_path.open("w") as fw:
        for raw in fr:
            line = json.loads(raw)
            record = {
                "instruction": INSTRUCTION,
                "context": line.get("sentence", ""),
                "response": "True" if line.get("relation") == "ADE_TRUE" else "False",
                "category": "classification"
            }
            fw.write(json.dumps(record))
            fw.write("\n")

if __name__ == "__main__":
    convert(TRAIN_TOPN, KNN_TRAIN)
    convert(TEST_TOPN, KNN_TEST)