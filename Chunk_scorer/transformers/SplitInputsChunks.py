import torch
import transformers
import logging
import sys
# from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, LlamaTokenizer
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import softmax

# Setup logging to stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
# model_id = '/scratch/ahcie-gpu2/openllama-models/MedLLaMA_13B'
# model_id = '/scratch/ahcie-gpu2/openllama-models/open_llama_7b_v2'
model_id="/home/users/riah/models/Llama-2-13b-hf"

# tokenizer.add_special_tokens({'end_token': '[END]'})

# ids = torch.tensor([    0, 13866,   338,   385, 15278,   393, 16612,   263,  3414, 29892,
#           3300,  2859,   411,   385,  1881,   393,  8128,  4340,  3030, 29889,
#          14350,   263,  2933,   393,  7128,  2486,  1614,  2167,   278,  2009,
#          29889,    13,    13,  2277, 29937,  2799,  4080, 29901,    13,  3492,
#            526,   385, 15129, 21110,   391, 29889,   450,  3414,   338,   304,
#           8500,   278,  1556,  8018,  8220, 29892,   607,  1818,   367,   297,
#           6702, 15094,  4741,  2287, 29903,   742,   525, 15094, 23711, 13152,
#           1660, 29903,   742,   525, 29911,  1525,  1299, 29903,   742,   525,
#          29909,  4198, 13845, 29903,   742,   525, 23845, 17923, 29903, 29918,
#          29956, 13054,   742,   525,  1177, 29950,  8979,  1806, 29903,   742,
#            525,  8618, 14849, 27266,   742,   525,  3035, 16173,  9047,  1001,
#           3352, 29918,  4986,   742,   525,  8618, 23524, 29918,  9800,   742,
#            525, 29909, 23338, 13780, 29903,   742,   525, 15094, 29963,  3919,
#          29903,   742,   525,  4571, 10051,  6632,  1660, 29903,   742,   525,
#           3217,  5746, 24306, 29918, 29956, 13054,   742,   525, 22933, 29949,
#           8426,  3040, 29928, 29918, 29956, 13054,   742,   525, 23711, 29934,
#           4897,  9375,   742,   525,  5454, 17171, 29903,   742,   525, 21514,
#          27888,  1299,  2890,   742,   525, 14816,  3580,  4986, 29924, 29918,
#           9800,   742,   525,  3970,  2890, 29918, 12256, 29918, 29911,  1525,
#           1299,   742,   525,  1254,  7833, 13309,  1299,  2890,   742,   525,
#          27616,  6545, 29923,  1254,  8098, 29918,  9800,   742,   525, 17171,
#          29903,  5477,   363,   278,  2183, 10541, 29889, 29871,  1222,  9422,
#          29901,   259, 29896, 29889, 15228, 29901,   301, 29899,   294,   862,
#          26584,   559, 29871,  5802,   491,   278,   364, 14170,   385, 29874,
#            261, 16945, 29871,   325,   747,  5378,  8348,   262,  6352,   267,
#            869,  5103, 29901,  8618, 14849, 27266, 29871, 29906, 29889, 15228,
#          26254,  1836, 13291, 29901,  6571, 29989,  2677, 29901, 19253,   310,
#           7067,  3090, 16103,   778,   274,   406,  1730, 23395, 29889, 13291,
#          29901,  4810,  5746, 24306, 29918, 29956, 13054, 29989,  2677, 29901,
#            310,   285,  4626,   262,   316,  5105,   362,  9316, 13291, 29901,
#          13756, 23524, 29918,  9800, 29989,  2677, 29901, 19253,   310,  7067,
#           3090, 16103,   778,   274,   406,  1730, 23395, 29889, 13291, 29901,
#           4810,  5746, 24306, 29918, 29956, 13054, 15228, 29901,   310,   285,
#           4626,   262,   316,  5105,   362,  9316, 13291, 29901, 13756, 23524,
#          29918,  9800,    13,    13,  2277, 29937, 10567, 29901, 29871,    13,
#           1545,  1070, 16810, 29882,  7492,  1788,   363, 10768,   284,   289,
#            650,   322, 29871, 14002, 16920, 29871,  1156, 21622,   272, 29871,
#            620,  9739,   869,    13,    13,  2277, 29937, 13291, 29901, 29871,
#             13, 15094,  4741,  2287, 29903])


# print( tokenizer.decode(ids))




 
class Get_Chunks_Inputs:
    def __init__(self): 
        self.tokenizer = LlamaTokenizer.from_pretrained(model_id)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


        
    def chuck_to_id(self,entity_dict):
        e = {}

        f = open(entity_dict, 'r')
        i=-1
        for line in f:
            i=i+1
            line = line.strip().split('\t')
            # ent_id = int(line[0])
            ent_name = line[0]
            e[ent_name] = i
        f.close()


        return e


    def prepare_embeddings(self,embedding_dict):
        entity2idx = {}
        idx2entity = {}
        i = 0
        embedding_matrix = []
        for key, entity in embedding_dict.items():
            entity2idx[key.strip()] = i
            idx2entity[i] = key.strip()
            i += 1
            embedding_matrix.append(entity)
        return entity2idx, idx2entity, embedding_matrix

    def get_mapping_and_matrix(self):
        entity_dict = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_DDI_IMPROVED/all_candiate_chunks_5_ddi.txt"
        entity_path = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_DDI_IMPROVED/all_posible_chuck_vector_5_ddi_medllama_13b_merged.npy"
        chunks_matrix = np.load(entity_path)
        mapping = self.chuck_to_id(entity_dict)
        return mapping,chunks_matrix
    
    def redefine_more_ipnuts(self,ids):
      
        source_sentence=self.tokenizer.decode(ids)
        logging.info(f"[SPLIT DEBUG] Decoded {len(ids)} tokens to {len(source_sentence)} chars")
        logging.info(f"[SPLIT DEBUG] First 300 chars: {source_sentence[:300]}")
        
        # Check if the required markers exist
        if "### Instruction:" not in source_sentence:
            logging.error(f"[SPLIT ERROR] Missing '### Instruction:' marker in decoded text")
            logging.error(f"Decoded text preview: {source_sentence[:500]}...")
            return []
        
        # Check for either "### Input: " or "### Input:\n" format
        if "### Input:" not in source_sentence:
            logging.error(f"[SPLIT ERROR] Missing '### Input:' marker in decoded text")
            logging.error(f"Decoded text preview: {source_sentence[:500]}...")
            return []
        
        try:
            before_instruction=source_sentence.split("### Instruction:")[0]
            # Handle both "### Input: " and "### Input:\n" formats
            if "### Input: \n" in source_sentence:
                after_input=source_sentence.split("### Instruction:")[1].split("### Input: \n")[1]
                instruction=source_sentence.split("### Instruction:")[1].split("### Input: \n")[0].strip()
            else:
                after_input=source_sentence.split("### Instruction:")[1].split("### Input:")[1]
                instruction=source_sentence.split("### Instruction:")[1].split("### Input:")[0].strip()
            before_Excamples=instruction.split(" Examples:")[0]
            excamples=instruction.split(" Examples:")[1].split("|")
        except IndexError as e:
            logging.error(f"[SPLIT ERROR] Failed to parse instruction format: {e}")
            logging.error(f"Source sentence preview: {source_sentence[:300]}...")
            import traceback
            traceback.print_exc()
            return []

        Inputs=[]
        logging.info(f"[SPLIT INPUTS] Found {len(excamples)} examples/chunks to process")
        
        for idx, excample in enumerate(excamples):
            I=before_instruction+"### Instruction:"+"\n"+before_Excamples+"Examples: "+ excample+"\n\n"+"### Input:"+after_input
            
            # Log chunk content
            if idx == 0 or idx == len(excamples)-1:  # Show first and last to reduce noise
                logging.info(f"[SPLIT INPUTS] Chunk {idx+1}/{len(excamples)}: {excample[:100]}...")
            
            encoderd_I=self.tokenizer(I)
            dic_={}
            dic_["input_ids"]=torch.tensor([encoderd_I["input_ids"]]).cuda()
            dic_["attention_mask"]=torch.tensor([encoderd_I["attention_mask"]]).cuda()
            dic_["labels"]=torch.tensor([encoderd_I["input_ids"]]).cuda()
            Inputs.append(dic_)
        
        logging.info(f"[SPLIT INPUTS] Returning {len(Inputs)} processed chunks")
        return Inputs
          
