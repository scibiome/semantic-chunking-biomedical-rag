import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from peft import PeftModel
import json

number_="8000"

checkpoint="checkpoint-%s"%number_

# SCORED S-HRT (3 examples): Requires model trained on *_s_hrt3.json (3 examples).
# Current scored model is trained on 1-example data → use chuck5_generation_scored_s_hrt_8000.py instead.
input_test_file_name="../2_relation_data_to_triple_train_data/test_triplet_chunk_scorer_scored_s_hrt3.json"
save_file_name="test_triplet_chunk_scorer_scored_s_hrt3_%s.json"%number_



lora_weights="../3_trainning_triple_model/TE_llama2_13b_ourmethod_chuck5_right_0_ChunkScorer_scored_s_hrt_1024"+"/"+checkpoint  

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model="/home/users/riah/models/Llama-2-13b-hf"

tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map='auto')

model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )



def make_inference(instruction, context = None):
  if context:
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
  else:
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: \n{instruction}\n\n### Response: \n"
  device = next(model.parameters()).device
  inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, max_length=1500).to(device)
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
      # Extract only the generated part (after the prompt)
      generated = text[len(prompt):].strip()
      return generated



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
        
        print(f"Sample {i+1}: {result[:100]}...")
        dic={}
        dic["instruction"]=instruction
        dic["context"]=sentence
        dic["ground_truth"]=ground_truth
        dic["response"]=result
        print(i+1)
        i=i+1
        fw.write(json.dumps(dic))
        fw.write("\n")
        fw.flush()

    print(f"Generation completed! Processed {i} samples.")



