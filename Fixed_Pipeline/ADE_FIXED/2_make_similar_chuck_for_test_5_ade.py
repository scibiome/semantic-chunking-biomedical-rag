import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig

MODEL_ID = "/home/users/riah/models/MedLLaMA_13B"
DATA_FILE = "/home/users/riah/project_thesis/BIoMedRAG/dataset/ade_classification/ade_corpus_v2_test.json"
STORED_CHUNKS_FILE = "stored_chucks_with_relation_ade_5.txt"
RETRIEVED_CHUNKS_FILE = "retrived_chucks_ade_5.npy"
LABEL_METADATA_FILE = "label_metadata_ade.json" 
OUTPUT_FILE = "test_data_chucks_ade_5.json"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)


def cos_similisty(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)


def make_text_embeddings(description):
    input_ids = tokenizer.encode(description, return_tensors="pt")
    output = model(input_ids, output_hidden_states=True)
    hidden_states = output.hidden_states
    input_embeddings = hidden_states[0]
    input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)
    final_embeddings = input_embeddings.detach().numpy()
    return final_embeddings


def load_data_base_relation_chuck():
    chucks = []
    relations = []
    with open(STORED_CHUNKS_FILE, "r") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            parts = line.split("--##--")
            if len(parts) != 2:
                continue
            chucks.append(parts[0])
            relations.append(parts[1])
    return chucks, relations


def load_label_metadata():
    id_to_label = {}
    with open(LABEL_METADATA_FILE, "r") as fr:
        metadata = json.load(fr)
        for entry in metadata:
            label_id = str(entry["id"])
            label_name = entry["label"]
            id_to_label[label_id] = label_name
    return id_to_label


relation_chunks, relation_chunks_with_r = load_data_base_relation_chuck()
relation_chunks_embeddings = np.load(RETRIEVED_CHUNKS_FILE)
LABEL_ID_TO_NAME = load_label_metadata()


def make_train_chunks(one_chuck_n_words):
    with open(OUTPUT_FILE, "w") as fw, open(DATA_FILE, "r") as fr:
        for idx, raw_line in enumerate(fr):
            if idx % 100 == 0:
                print(idx)

            raw_line = raw_line.strip()
            if not raw_line:
                continue
            line = json.loads(raw_line)

            tokens = [tok for tok in line.get("text", "").split(" ") if tok]
            if len(tokens) > one_chuck_n_words:
                chunk_size = len(tokens) / one_chuck_n_words
                chunked_data = np.array_split(tokens, chunk_size)
            else:
                chunked_data = [tokens]

            all_chucks = []
            for chunk in chunked_data:
                chunk = " ".join(chunk)
                chunk_embedding = make_text_embeddings(chunk)

                cos_scores = [
                    cos_similisty(chunk_embedding, relation_chunks_embeddings[i])
                    for i in range(len(relation_chunks_embeddings))
                ]

                top_5_idx = np.argsort(cos_scores)[-5:]
                top_5_chucks = [relation_chunks[i] for i in top_5_idx]
                top_5_relations = [relation_chunks_with_r[i] for i in top_5_idx]

                all_chucks.append({chunk: [top_5_chucks, top_5_relations]})

            relation_value = str(line.get("label"))
            relation_label = LABEL_ID_TO_NAME.get(relation_value, relation_value)

            record = {
                "relation": relation_label,
                "sentence": " ".join(tokens),
                "all_chucks": all_chucks,
            }
            fw.write(json.dumps(record))
            fw.write("\n")


if __name__ == "__main__":
    make_train_chunks(one_chuck_n_words=5)
