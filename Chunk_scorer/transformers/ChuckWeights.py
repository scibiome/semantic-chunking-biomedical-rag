import torch
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
import  torch
import copy
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.functional import softmax

from torch.autograd import Variable

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





class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out,weight = self.attention(x)

        out = self.feed_forward(out)
        return out,weight


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):

        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        # print("mm",attention.size())
        context = torch.matmul(attention, V)
        return context,attention


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
       
       
        scale = K.size(-1) ** -0.5
        context,weight = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out,weight

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


 
class Chuck_weight:
    def __init__(self, device: str = "cpu"): 
        self.dim_model=5120
        self.num_head=1
        self.hidden=3072
        self.dropout=0.5
        self.num_encoder=6
        self.device = torch.device(device)
        self.encoder = Encoder(self.dim_model, self.num_head, self.hidden, self.dropout).to(self.device)
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


    def get_weight_value(self,ids):
        decoded = self.tokenizer.decode(ids)
        # Get the instruction part (between ### Instruction: and ### Input:)
        instruction_part = decoded.split("### Instruction:")[1].split("### Input: ")[0].strip()
        
        # The instruction format is: [base instruction with examples including '|' characters]|chunk1|chunk2|...
        # We need to find where the actual chunks start after the examples
        # Chunks are in the format: "Context: ... Response: ..."
        
        # Find the split point - look for the last name|name|name pattern before the chunks
        # Split by | and process backwards
        parts = instruction_part.split("|")
        
        # The chunks are the parts that contain "Context:" at the start
        # The last part before chunks should end with a triple pattern (head|relation|tail)
        chunks_only = []
        for i in range(len(parts)-1, -1, -1):
            part = parts[i].strip()
            # If this part starts with "Context:", it's a chunk
            if part.startswith("Context:"):
                chunks_only.append(part)
            else:
                # Once we hit a non-chunk part, stop
                break
        
        # Reverse to get correct order
        source_sentence = list(reversed(chunks_only))
        
        # Get the context (input sentence)
        source_context = decoded.split("### Input:")[1].split("### Response:")[0].strip()
        
        # Debug: print what we extracted
        logging.info(f"DEBUG: Extracted {len(source_sentence)} chunks")
        if len(source_sentence) > 0:
            logging.info(f"DEBUG: First chunk: {source_sentence[0][:100]}")
        logging.info(f"DEBUG: Context: {source_context}")
        
        # Combine context and chunks
        source_sentence = [source_context] + source_sentence if source_sentence else [source_context]
        mapping,chunks_matrix=self.get_mapping_and_matrix()

        chuck_embedding=[]
        missing_chunks = []
        matrix_size = chunks_matrix.shape[0]  # Get actual matrix size
        for chunk in source_sentence:
            if chunk in mapping:
                chunk_idx = mapping[chunk]
                # Bounds check: ensure index is within matrix size
                if chunk_idx < matrix_size:
                    chuck_embedding.append(chunks_matrix[chunk_idx])
                else:
                    missing_chunks.append(chunk)
                    logging.warning(f"WARNING: Chunk index {chunk_idx} out of bounds (matrix size: {matrix_size}): {chunk[:100]}...")
            else:
                missing_chunks.append(chunk)
                logging.warning(f"WARNING: Chunk not found in embedding matrix: {chunk[:100]}...")

        if len(missing_chunks) > 0:
            logging.warning(f"[CHUNK WEIGHTS] Skipping {len(missing_chunks)} chunks not in embedding matrix")
            logging.warning(f"[CHUNK WEIGHTS] Processing {len(chuck_embedding)} chunks with embeddings")

        if len(chuck_embedding) == 0:
            logging.error("[CHUNK WEIGHTS] ERROR: No chunks available, creating dummy embedding")
            # Create a dummy embedding with correct dimensions
            dummy_embedding = np.zeros((1, chunks_matrix.shape[1]))
            chuck_embedding.append(dummy_embedding)

        # Convert to numpy array first for efficiency, then to tensor
        arr = np.asarray(chuck_embedding, dtype=np.float32)  # list -> single ndarray (fast)
        Chucks_embedding = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # requires_grad=False


        chucks_feather,weight=self.encoder(Chucks_embedding)
        #   print(weight.size())
        return self.encoder, weight

