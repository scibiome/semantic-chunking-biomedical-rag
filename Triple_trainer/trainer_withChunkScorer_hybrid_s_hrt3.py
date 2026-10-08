import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlamaTokenizer
from trl import SFTTrainer
from datasets import load_dataset


def formatting_func(example):
    if example.get("context", "") != "":
        input_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Input: \n"
            f"{example['context']}\n\n"
            "### Response: \n"
            f"{example['response']}"
        )
    else:
        input_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            "### Response:\n"
            f"{example['response']}"
        )

    return {"text": input_prompt}


def prepare_data(path):
    data = load_dataset("json", data_files=path)
    formatted_data = data.map(formatting_func)
    return formatted_data["train"]


# HYBRID CHUNKING + S-HRT (3 examples per instruction)
train_path = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/train_triplet_chunk_scorer_hybrid_s_hrt3.json"
test_path = "/home/users/riah/project_thesis/BIoMedRAG/2_relation_data_to_triple_train_data/test_triplet_chunk_scorer_hybrid_s_hrt3.json"
train = prepare_data(train_path)
test = prepare_data(test_path)

model_id = "/home/users/riah/models/Llama-2-13b-hf"

qlora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
base_model.resize_token_embeddings(len(tokenizer))

supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=train,
    eval_dataset=test,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=10000,
        save_steps=1000,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        output_dir="TE_llama2_13b_ourmethod_chuck5_right_0_ChunkScorer_hybrid_s_hrt3_1024",
        optim="paged_adamw_8bit",
        fp16=True,
    ),
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="text",
    max_seq_length=1024,
)

supervised_finetuning_trainer.train()


