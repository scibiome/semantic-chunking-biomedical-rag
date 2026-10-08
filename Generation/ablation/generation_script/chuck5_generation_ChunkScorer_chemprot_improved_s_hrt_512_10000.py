import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from peft import PeftModel
import json


number_ = "10000"
checkpoint = f"checkpoint-{number_}"

# ChemProt IMPROVED + chunk scorer + S-HRT (1 example), 512 training
input_test_file_name = (
    "../2_relation_data_to_triple_train_data/"
    "test_triplet_chunk_scorer_chemprot_improved_s_hrt.json"
)
save_file_name = (
    f"chuck_5_triplet_sim_abaltion_ChunkScorer_chemprot_improved_s_hrt_512_{number_}.json"
)

# LoRA checkpoint from ChemProt IMPROVED S-HRT 512 training
lora_weights = (
    "../3_trainning_triple_model/"
    "TE_llama2_13b_ourmethod_chuck5_right_0_ChunkScorer_chemprot_improved_s_hrt_512/"
    + checkpoint
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model_path = "/home/users/riah/models/Llama-2-13b-hf"

tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
)


def make_inference(instruction, context=None):
    if context:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
            "### Instruction: \n"
            f"{instruction}\n\n"
            "### Input: \n"
            f"{context}\n\n"
            "### Response: \n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction: \n"
            f"{instruction}\n\n"
            "### Response: \n"
        )

    device = next(model.parameters()).device
    # Note: training max_seq_length was 512, but as in existing scripts
    # we allow up to ~1500 tokens for the prompt; inputs are typically shorter.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=1500,
        truncation=True,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text


if __name__ == "__main__":
    with open(save_file_name, "w") as fw:
        i = 0
        with open(input_test_file_name, "r", encoding="utf-8") as fr:
            for line in fr:
                obj = json.loads(line.strip())
                instruction = obj["instruction"]
                sentence = obj["context"]
                ground_truth = obj["response"]
                predicted = make_inference(instruction, sentence)

                record = {
                    "sentence": sentence,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                }
                print(i)
                i += 1
                fw.write(json.dumps(record))
                fw.write("\n")
                fw.flush()

    print("Generation completed!")




