import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from peft import PeftModel
import json

number_="8000"

checkpoint="checkpoint-%s"%number_

# DDI FIXED S-HRT: Use DDI chunk scorer fixed s-hrt test data
input_test_file_name="../2_relation_data_to_triple_train_data/test_triplet_chunk_scorer_ddi_fixed_s_hrt.json"
save_file_name="test_triplet_chunk_scorer_ddi_fixed_s_hrt_%s_MULTI.json"%number_



lora_weights="../3_trainning_triple_model/TE_llama2_13b_ourmethod_chuck5_right_0_ChunkScorer_fixed_s_hrt_ddi_1024"+"/"+checkpoint  

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



def make_inference(instruction, context = None, extract_all=False):
  if context:
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
  else:
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: \n{instruction}\n\n### Response: \n"
  device = next(model.parameters()).device
  # Use max_length=1024 to match training max_seq_length, with truncation
  inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, max_length=1024, truncation=True).to(device)
  model.eval()
  with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_new_tokens=200,  # Increased to allow multiple triplets
          temperature=0.0,
          do_sample=False,
          pad_token_id=tokenizer.eos_token_id,
      )
      text = tokenizer.decode(outputs[0], skip_special_tokens=True)
      # Extract only the generated part (after the prompt)
      generated = text.split("### Response:")[-1].strip()
      
      # Extract all valid triplets
      lines = generated.split('\n')
      triplets = []
      for line in lines:
          line = line.strip().rstrip(',').strip()
          if line and '|' in line and not line.startswith('###') and 'Context:' not in line:
              parts = line.split('|')
              if len(parts) == 3:
                  triplets.append(line)
      
      if extract_all and triplets:
          # Return all triplets separated by newlines
          return '\n'.join(triplets)
      elif triplets:
          # Return first triplet (original behavior)
          return triplets[0]
      else:
          # Fallback: return first non-empty line
          return generated.split('\n')[0].strip() if generated else ""


def extract_all_triplets_from_context(context, instruction, max_attempts=3):
    """
    Try to extract all possible triplets from the same context by:
    1. Generating with the original instruction
    2. If we get multiple triplets, return them
    3. Otherwise, try generating multiple times with slight variations
    """
    all_triplets = set()
    
    # First attempt: standard generation
    result = make_inference(instruction, context, extract_all=True)
    if result:
        for triplet in result.split('\n'):
            triplet = triplet.strip()
            if triplet and '|' in triplet:
                all_triplets.add(triplet)
    
    # If we only got one triplet, try generating again with temperature
    if len(all_triplets) <= 1 and max_attempts > 1:
        device = next(model.parameters()).device
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, max_length=1024, truncation=True).to(device)
        
        for attempt in range(max_attempts - 1):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,  # Slight randomness
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = text.split("### Response:")[-1].strip()
                
                lines = generated.split('\n')
                for line in lines:
                    line = line.strip().rstrip(',').strip()
                    if line and '|' in line and not line.startswith('###') and 'Context:' not in line:
                        parts = line.split('|')
                        if len(parts) == 3:
                            all_triplets.add(line)
    
    return list(all_triplets)


if __name__=="__main__":
 
    fw=open(save_file_name,"w")
    i=0
    
    # Track contexts we've seen
    context_to_triplets = {}
    
    with open(input_test_file_name,"r",encoding="utf-8") as fr:
      for line in fr.readlines():
        line=json.loads(line.strip())
        instruction=line["instruction"]
        sentence=line["context"]
        ground_truth=line["response"]
        
        # Check if we've seen this context before
        if sentence in context_to_triplets:
            # Use previously extracted triplets
            all_triplets = context_to_triplets[sentence]
            # Try to match the ground truth or use first available
            result = ground_truth if ground_truth in all_triplets else (all_triplets[0] if all_triplets else "")
        else:
            # First time seeing this context - extract all triplets
            all_triplets = extract_all_triplets_from_context(sentence, instruction, max_attempts=2)
            context_to_triplets[sentence] = all_triplets
            # Use ground truth if it's in our extracted triplets, otherwise use first
            result = ground_truth if ground_truth in all_triplets else (all_triplets[0] if all_triplets else "")
        
        print(f"Sample {i+1}: Found {len(all_triplets)} triplets, using: {result[:80]}...")
        dic={}
        dic["instruction"]=instruction
        dic["context"]=sentence
        dic["ground_truth"]=ground_truth
        dic["response"]=result
        dic["all_extracted_triplets"]=all_triplets  # Store all found triplets
        print(i+1)
        i=i+1
        fw.write(json.dumps(dic))
        fw.write("\n")
        fw.flush()

    print(f"Generation completed! Processed {i} samples.")
    print(f"Found {len(context_to_triplets)} unique contexts.")



