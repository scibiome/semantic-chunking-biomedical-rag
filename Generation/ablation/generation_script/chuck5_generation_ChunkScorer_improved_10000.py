import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from peft import PeftModel
import json
import sys
import os

# Add current directory to path to import entity_swap_detection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entity_swap_detection import (
    detect_and_fix_entity_swap
)

number_="10000"

checkpoint="checkpoint-%s"%number_

# ChunkScorer IMPROVED: Use improved test data with chunk scorer reranking
input_test_file_name="../2_relation_data_to_triple_train_data/test_triplet_chunk_scorer_improved.json"
save_file_name="chuck_5_triplet_sim_abaltion_ChunkScorer_improved_entity_swap_%s.json"%number_


lora_weights="../3_trainning_triple_model/TE_llama2_13b_ourmethod_chuck5_right_0_ChunkScorer_improved"+"/"+checkpoint  

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
  # Added truncation=True and increased max_length
  inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, max_length=2000, truncation=True).to(device)
  model.eval()
  with torch.no_grad():
      # Increased max_new_tokens from 50 to 120 to prevent truncation
      # Added stopping criteria to prevent instruction repetition
      outputs = model.generate(
          **inputs,
          max_new_tokens=120,
          do_sample=False,  # Greedy decoding for consistency
          pad_token_id=tokenizer.eos_token_id,
          eos_token_id=tokenizer.eos_token_id,
          # Add repetition penalty to reduce repetitive patterns
          repetition_penalty=1.1,  # Slight penalty to avoid repetition
          # Add length penalty to prefer shorter, more concise responses
          length_penalty=0.8,  # Slightly prefer shorter responses
          # Early stopping when EOS is found
          early_stopping=True,
      )
      results = tokenizer.decode(outputs[0], skip_special_tokens=True)
      
      # IMPROVED Post-processing: Extract LAST valid triplet (best performing approach - 77.50% F1)
      if "### Response:" in results:
          # Find all occurrences of "### Response:"
          response_sections = results.split("### Response:")
          if len(response_sections) > 1:
              # Find the LAST valid triplet (most refined answer)
              last_valid_response = None
              
              # Iterate from last to first to find the last valid triplet
              for i in range(len(response_sections) - 1, 0, -1):
                  response_part = response_sections[i].strip()
                  
                  # Stop at "### Input:" or "### Instruction:" if they appear
                  for stop_marker in ["### Input:", "### Instruction:"]:
                      if stop_marker in response_part:
                          response_part = response_part.split(stop_marker)[0].strip()
                  
                  # Check if this response contains a valid triplet
                  lines = response_part.split("\n")
                  for line in lines:
                      line = line.strip().rstrip(',').strip()
                      if line and "|" in line and not line.startswith("###"):
                          parts = line.split("|")
                          if len(parts) >= 3:
                              # Valid triplet found - take first 3 parts only (handle invalid format)
                              triplet_str = "|".join(parts[:3])
                              last_valid_response = triplet_str
                              break
                  
                  if last_valid_response:
                      # Additional check: detect and fix entity swap if needed
                      sentence = context if context else ""
                      parts = last_valid_response.split("|")
                      relation = parts[1].strip() if len(parts) >= 2 else ""
                      corrected_triplet, was_swapped = detect_and_fix_entity_swap(
                          sentence,
                          last_valid_response,
                          relation
                      )
                      
                      # Reconstruct with last valid response (corrected if swapped)
                      before_response = "### Response:".join(response_sections[:i])
                      results = before_response + "### Response:\n" + corrected_triplet
                      break
              
              # Fallback: if no valid triplet found, use last response section
              if not last_valid_response:
                  last_response = response_sections[-1].strip()
                  for stop_marker in ["### Input:", "### Instruction:"]:
                      if stop_marker in last_response:
                          last_response = last_response.split(stop_marker)[0].strip()
                  before_response = "### Response:".join(response_sections[:-1])
                  if before_response:
                      results = before_response + "### Response:\n" + last_response
                  else:
                      results = "### Response:\n" + last_response
      
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
        predicted=make_inference(instruction,sentence)
        
        # Use ORIGINAL format - no post-processing (same as improved_512)!
        Dic_={}
        Dic_["sentence"]=sentence
        Dic_["ground_truth"]=ground_truth
        Dic_["predicted"]=predicted
        print(i)
        i=i+1
        fw.write(json.dumps(Dic_))
        fw.write("\n")
        fw.flush()

    print("Generation completed!")




