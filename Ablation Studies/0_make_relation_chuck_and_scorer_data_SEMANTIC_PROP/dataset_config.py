"""
Dataset-specific paths and config for component analysis (GM-CIHT, ChemProt, DDI).
Use environment variable DATASET=gmciht|chemprot|ddi to select dataset.
Default: gmciht (GM-CIHT).
"""

import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SCRIPT_DIR.parent  # BIoMedRAG

DATASETS = {
    "gmciht": {
        "dataset_dir": _BASE_DIR / "dataset" / "0_GM-CIHT",
        "dev_file": "renew_triplet_T_dev.jsonl",
        "train_file": "renew_triplet_T_train.jsonl",
        "test_file": "renew_triplet_T_test_deduplicated.jsonl",
        "config_file": "configs/config.json",
        "relation_def_file": "relation_with_defination.txt",
        "relation_embedding_file": "relation_description.npy",
        "output_suffix": "semantic_prop",
        "knn_train": "../2_relation_data_to_triple_train_data/KNN_demo_train.json",
        "knn_test": "../2_relation_data_to_triple_train_data/KNN_demo_test.json",
    },
    "chemprot": {
        "dataset_dir": _BASE_DIR / "dataset" / "2_chemprot",
        "dev_file": "dev_triples_gmciht_format.jsonl",
        "train_file": "train_triples_gmciht_format.jsonl",
        "test_file": "test_triples_gmciht_format.jsonl",
        "config_file": str(_BASE_DIR / "0_make_relation_chuck_and_scorer_data_IMPROVED" / "configs" / "chemprot_config.json"),
        # Reuse relation files from full ChemProt hybrid pipeline (CHEMPROT_IMPROVED)
        "relation_def_file": str(_BASE_DIR / "0_make_relation_chuck_and_scorer_data_CHEMPROT_IMPROVED" / "relation_with_defination.txt"),
        "relation_embedding_file": str(_BASE_DIR / "0_make_relation_chuck_and_scorer_data_CHEMPROT_IMPROVED" / "relation_description_chemprot.npy"),
        "output_suffix": "semantic_prop_chemprot",
        "knn_train": "../2_relation_data_to_triple_train_data/KNN_chemprot_train_SINGLE.json",
        "knn_test": "../2_relation_data_to_triple_train_data/KNN_chemprot_test_SINGLE.json",
    },
    "ddi": {
        "dataset_dir": _BASE_DIR / "dataset" / "1_DDI",
        "dev_file": "dev_gmciht_format_no_none.jsonl",
        "train_file": "train_gmciht_format_no_none.jsonl",
        "test_file": "test_gmciht_format_no_none.jsonl",
        "config_file": str(_BASE_DIR / "0_make_relation_chuck_and_scorer_data_IMPROVED" / "configs" / "ddi_config.json"),
        # Reuse relation files from full DDI hybrid pipeline (DDI_FIXED)
        "relation_def_file": str(_BASE_DIR / "0_make_relation_chuck_and_scorer_data_DDI_FIXED" / "relation_with_defination.txt"),
        "relation_embedding_file": str(_BASE_DIR / "0_make_relation_chuck_and_scorer_data_DDI_FIXED" / "relation_description.npy"),
        "output_suffix": "semantic_prop_ddi",
        "knn_train": "../2_relation_data_to_triple_train_data/KNN_ddi_train_SINGLE.json",
        "knn_test": "../2_relation_data_to_triple_train_data/KNN_ddi_test_SINGLE.json",
    },
}


def get_dataset_name():
    """Return dataset key from env (gmciht, chemprot, ddi)."""
    name = os.environ.get("DATASET", "gmciht").strip().lower()
    if name not in DATASETS:
        raise ValueError(f"Unknown DATASET={name}. Use one of: {list(DATASETS.keys())}")
    return name


def get_dataset_settings(dataset_name=None):
    """Return path/config dict for the given dataset (default from DATASET env)."""
    if dataset_name is None:
        dataset_name = get_dataset_name()
    s = DATASETS[dataset_name].copy()
    # Resolve paths relative to SEMANTIC_PROP dir
    s["dataset_dir"] = s["dataset_dir"]
    s["input_dev"] = str(s["dataset_dir"] / s["dev_file"])
    s["input_train"] = str(s["dataset_dir"] / s["train_file"])
    s["input_test"] = str(s["dataset_dir"] / s["test_file"])
    if not os.path.isabs(s["config_file"]):
        s["config_file"] = str(_SCRIPT_DIR / s["config_file"])
    if not os.path.isabs(s["relation_def_file"]):
        s["relation_def_file"] = str(_SCRIPT_DIR / s["relation_def_file"])
    if not os.path.isabs(s["relation_embedding_file"]):
        s["relation_embedding_file"] = str(_SCRIPT_DIR / s["relation_embedding_file"])
    s["knn_train"] = str(_SCRIPT_DIR / s["knn_train"])
    s["knn_test"] = str(_SCRIPT_DIR / s["knn_test"])
    s["output_dir"] = str(_SCRIPT_DIR / "output")
    return s
