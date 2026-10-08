
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np

model_id = '/home/users/riah/models/MedLLaMA_13B'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

all_vectors = []
with open("relation_with_defination.txt", "r") as fr:
    for line in fr.readlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        text = parts[1] if len(parts) > 1 else parts[0]

        input_ids = tokenizer.encode(text, return_tensors='pt')
        output = model(input_ids, output_hidden_states=True)
        hidden_states = output.hidden_states
        input_embeddings = hidden_states[0]
        input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)
        final_embeddings = input_embeddings.detach().cpu().numpy()
        all_vectors.append(final_embeddings)

np.save("relation_description.npy", np.array(all_vectors))



