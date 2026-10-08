import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# SEMANTIC + PROPOSITION VARIANT (Ablation)
number_="10000"
checkpoint="checkpoint-%s"%number_

# Paths
input_test_file_name="../2_relation_data_to_triple_train_data/test_triplet_chunk_scorer_semantic_prop_s_hrt.json"
lora_weights="../3_trainning_triple_model/TE_llama2_13b_semantic_prop_s_hrt"+"/"+checkpoint
save_file_name="chuck_5_triplet_sim_abaltion_semantic_prop_10000.json"

model_id = "/home/users/riah/models/Llama-2-13b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map='auto',
)

model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
    device_map='auto',
)

def make_inference(instruction, context = None):
  if context is not None:
      prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{instruction}\n\n"
      f"### Input: \n"
      f"{context}\n\n"
      f"### Response: \n")
  else:
      prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{instruction}\n\n"
      f"### Response:\n")
  
  inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:0")
  
  with torch.no_grad():
      outputs = model.generate(**inputs, max_new_tokens=100)
      
      results=tokenizer.decode(outputs[0], skip_special_tokens=True)
      return results

if __name__=="__main__":
 
    fw=open(save_file_name,"w")
    i=0
    with open(input_test_file_name,"r",encoding="utf-8") as fr:
      for line in fr.readlines():
        line=json.loads(line.strip())
        instruction=line["instruction"]
        sentence=line["context"]
        ground_truth=line["response"]
        result=make_inference(instruction,sentence)
        
        result=result.split("### Response: ")
        result=result[1].strip()
        
        print(result)
        dic={}
        dic["instruction"]=instruction
        dic["context"]=sentence
        dic["ground_truth"]=ground_truth
        dic["response"]=result
        print(i)
        i=i+1
        fw.write(json.dumps(dic))
        fw.write("\n")
        fw.flush()

    print("Generation completed!")

