import json
from itertools import permutations
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
import numpy as np
import os
import sqlite3
from itertools import islice
# Align to original model for equal comparison with semantic
model_id = '/home/users/riah/models/Llama-2-13b-hf'
import json
# model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
# tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, torch_dtype=torch.float16, device_map='auto',
    trust_remote_code=True,
)


tokenizer = LlamaTokenizer.from_pretrained(model_id)
# Ensure pad token is set robustly (LLaMA often lacks a pad token)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>', 'pad_token': '[PAD]'})
    else:
        # Prefer using EOS as PAD for causal models
        tokenizer.pad_token = tokenizer.eos_token

# Fallback: if still missing, add explicit PAD
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'

# Resolve pad_token_id
if tokenizer.pad_token_id is None:
    pid = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if pid is None or pid == tokenizer.unk_token_id:
        # Ensure PAD is in vocab
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        pid = tokenizer.convert_tokens_to_ids('[PAD]')
    tokenizer.pad_token_id = pid

# Make model aware of PAD
try:
    model.config.pad_token_id = tokenizer.pad_token_id
except Exception:
    pass

# Ensure model can handle tokenizer's vocab size
try:
    model.resize_token_embeddings(len(tokenizer))
except Exception:
    pass

def make_chucks(items):
    all_chucks=[]
    for i in range(len(items)+1):
        chucks=[]
        for p in permutations(items,i):
            chucks.append(p)
        if i==len(items):
            all_chucks=all_chucks+[chucks[0]]
        else:
            all_chucks = all_chucks +chucks
    return all_chucks



def get_chucks(file):
    j=-1
  
    all_c=[]
    with open(file,"r") as fr:
        for line in fr.readlines():
            j=j+1
            # print(j)
            line=json.loads(line)
            topn_sim_chuck=line["topn_sim_chuck"]
            # print(topn_sim_chuck)
            all_lists_=[]
            for key, value in topn_sim_chuck.items():
                all_="Context: "+ key+" Response: "+ value
                all_lists_.append(all_)
            chucks_=make_chucks(all_lists_)
            # print(chucks_)
            for c in chucks_:
                if len(c) !=0:
                    c=" ".join(c)
                    all_c.append(c)
    return all_c




def get_topn_whole_sentence(file):
    j=-1
  
    all_c=[]
    with open(file,"r") as fr:
        for line in fr.readlines():
            j=j+1
            # print(j)
            line=json.loads(line)
            context=line["context"]
            instr_field = line.get("instruction", "")
            if "Examples:  " in instr_field:
                instruction = instr_field.split("Examples:  ", 1)[1]
            else:
                instruction = instr_field
            # print(instruction)
            all_c.append(instruction)
            all_c.append(context)


    return all_c




def _permutation_strings(items, max_total=64):
    """Yield up to max_total permutations per record.

    - Always include the first full-length permutation
    - Then include shorter-length permutations in increasing length order
    """
    all_lists_ = items
    n = len(all_lists_)
    emitted = 0

    # First full-length permutation
    if n > 0:
        first_full = next(permutations(all_lists_, n), None)
        if first_full:
            yield " ".join(first_full)
            emitted += 1
            if emitted >= max_total:
                return

    # Then shorter permutations starting from length 1 upwards
    for i in range(1, n):
        for p in permutations(all_lists_, i):
            if not p:
                continue
            yield " ".join(p)
            emitted += 1
            if emitted >= max_total:
                return


def _stream_candidates():
    # train/test candidates with permutations (capped for parity with semantic)
    for src in ("train_data_topn_chucks_5.json", "test_data_topn_chucks_5_deduplicated.json"):
        with open(src, "r") as fr:
            for line in fr:
                line = json.loads(line)
                topn_sim_chuck = line["topn_sim_chuck"]
                items = []
                for key, value in topn_sim_chuck.items():
                    items.append("Context: " + key + " Response: " + value)
                for s in _permutation_strings(items, max_total=64):
                    yield s
    # KNN context/instruction (no permutations; same as original)
    knn_train = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_train.json"
    knn_test = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/KNN_demo_test.json"
    for knn in (knn_train, knn_test):
        if os.path.exists(knn):
            with open(knn, "r") as fr:
                for line in fr:
                    line = json.loads(line)
                    context = line.get("context", "")
                    instr_field = line.get("instruction", "")
                    if "Examples:  " in instr_field:
                        instruction = instr_field.split("Examples:  ", 1)[1]
                    else:
                        instruction = instr_field
                    if instruction:
                        yield "Context: " + instruction
                    if context:
                        yield "Context: " + context


def _batched(it, size):
    it = iter(it)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def _embed_batch(texts):
    # Manual padding to avoid tokenizer padding errors
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Fallbacks if tokenizer pad id is still not set
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    encoded = [tokenizer.encode(t, add_special_tokens=True, truncation=True, max_length=256) for t in texts]
    if not encoded:
        return np.zeros((0, getattr(model.config, 'hidden_size', 4096)), dtype=np.float32)
    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=model.device)
    for i, ids in enumerate(encoded):
        if len(ids) > 0:
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=model.device)

    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)
        hs0 = output.hidden_states[0]
        attn = (input_ids != pad_id).unsqueeze(-1).to(hs0.dtype)
        summed = (hs0 * attn).sum(dim=1)
        lengths = attn.sum(dim=1).clamp(min=1)
        mean = summed / lengths
        return mean.detach().cpu().numpy()


def main():
    fw = open("all_candiate_chunks_5_new.txt", "w")
    # On-disk dedup to preserve global set() semantics without OOM
    conn = sqlite3.connect("candidates_seen_new.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS seen (text TEXT PRIMARY KEY)")
    conn.commit()

    shard_index = 0
    batch_size = 16  # Increased for faster processing
    total = 0
    try:
        batch = []
        for cand in _stream_candidates():
            try:
                cur.execute("INSERT INTO seen(text) VALUES (?)", (cand,))
                conn.commit()
            except sqlite3.IntegrityError:
                continue
            fw.write(cand)
            fw.write("\n")
            total += 1
            batch.append(cand)
            if len(batch) >= batch_size:
                embeds = _embed_batch(batch)
                np.save(f"all_posible_chuck_vector_5_medllama_13b_shard_new_{shard_index:05d}.npy", embeds)
                shard_index += 1
                batch.clear()
                print(total)
        if batch:
            embeds = _embed_batch(batch)
            np.save(f"all_posible_chuck_vector_5_medllama_13b_shard_new_{shard_index:05d}.npy", embeds)
            shard_index += 1
            batch.clear()
    finally:
        fw.close()
        conn.close()


if __name__ == "__main__":
    main()

"""
top2 137730
top1: 81
"""