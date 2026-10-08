"""
Hybrid Semantic + Sliding Window version of relation-chunk mapping
Uses HybridSemanticChunker for robust chunking of biomedical text
"""

import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import os

# Add parent directory to path to import chunker
sys.path.insert(0, os.path.dirname(__file__))
from improved_semantic_chunker import ImprovedSemanticChunker

# Model configuration
model_id = '/home/users/riah/models/MedLLaMA_13B'

print("Loading model...")
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
print("Model loaded successfully!")

# Initialize IMPROVED semantic chunker
print("Initializing Improved Semantic Chunker...")
semantic_chunker = ImprovedSemanticChunker(
    window_size=7,
    stride=3,
    min_chunk_size=5,
    max_chunk_size=15
)
print("Improved Chunker ready!")
print("Strategy: Semantic boundaries + overlapping windows")
print("  - Splits at commas, conjunctions, semicolons")
print("  - Applies sliding window within each segment")
print("  - Preserves single-word entities and terms")
print("  - No duplicate chunks guaranteed!")


def cos_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors"""
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return cosine_similarity


def relation_with_definition():
    """Load relation definitions"""
    RD = {}
    with open("relation_with_defination.txt", "r") as fr:
        i = -1
        for line in fr.readlines():
            i = i + 1
            line = line.strip().split("\t")
            RD[line[0]] = i
    return RD


R_D = relation_with_definition()
R_D_embedding_matrix = np.load("relation_description.npy")


def make_text_embeddings(description):
    """Generate embeddings for text using the model"""
    input_ids = tokenizer.encode(description, return_tensors='pt', truncation=True, max_length=512)
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)
        hidden_states = output.hidden_states
        input_embeddings = hidden_states[0]
        input_embeddings = torch.mean(input_embeddings.squeeze(0), dim=0)
        final_embeddings = input_embeddings.detach().cpu().numpy()
    
    return final_embeddings


def make_chunck_semantic(target_chunks=5):
    """
    Create semantic chunks for each relation
    Uses SemanticChunker instead of fixed-size chunking
    """
    fw_ = open("relation_with_chucks_5_improved.json", "w")
    relation_chucks = {}
    j = -1
    
    print("\nProcessing sentences from dataset...")
    with open("../dataset/0_GM-CIHT/renew_triplet_T_dev.jsonl", "r") as fr:
        for line in fr.readlines():
            j = j + 1
            if j % 10 == 0:
                print(f"Processing sentence {j}...")
            
            line = json.loads(line)
            relation = line["PREDICATE"]
            relation_embedding = R_D_embedding_matrix[R_D[relation]]
            
            # Get original sentence
            sentence = line["SENTENCE"]
            
            # IMPROVED CHUNKING: Entity preservation + relation context
            try:
                # Improved chunker with relation-aware chunking
                # - Entity boundary preservation (no splits)
                # - Relation context preservation (triggers with entities)
                # - Uncertainty detection (might, may, possibly)
                # - Clinical context awareness (TREATS vs AFFECTS)
                chunks = semantic_chunker.chunk_for_triplet_extraction(
                    sentence, 
                    relation=relation,
                    num_chunks=target_chunks
                )
                
                # Extract text from chunk dictionaries
                semantic_chunks = [c['text'] for c in chunks]
                
            except Exception as e:
                print(f"Warning: Improved chunking failed for sentence {j}: {e}")
                # Fallback: split sentence in half
                words = sentence.split()
                mid = len(words) // 2
                semantic_chunks = [
                    ' '.join(words[:mid]),
                    ' '.join(words[mid:])
                ] if len(words) > 6 else [sentence]
            
            # Calculate similarity of each chunk with the relation
            cos_similarities = []
            for chunk in semantic_chunks:
                if chunk.strip():  # Only process non-empty chunks
                    chunk_embedding = make_text_embeddings(chunk)
                    similarity = cos_similarity(relation_embedding, chunk_embedding)
                    cos_similarities.append((chunk, similarity))
            
            # Sort by similarity and take top chunks
            if len(cos_similarities) >= 2:
                # Sort by similarity (highest first)
                cos_similarities.sort(key=lambda x: x[1], reverse=True)
                # Take top 2 chunks
                top_chunks = [chunk for chunk, _ in cos_similarities[:2]]
                
                if relation not in relation_chucks:
                    relation_chucks[relation] = top_chunks
                else:
                    relation_chucks[relation] = relation_chucks[relation] + top_chunks
            else:
                # If we have fewer than 2 chunks, use what we have
                chunks_only = [chunk for chunk, _ in cos_similarities]
                if relation not in relation_chucks:
                    relation_chucks[relation] = chunks_only
                else:
                    relation_chucks[relation] = relation_chucks[relation] + chunks_only
    
    # Write results
    print(f"\nWriting results for {len(relation_chucks)} relations...")
    for relation, chucks in relation_chucks.items():
        DIc_ = {}
        DIc_[relation] = chucks
        fw_.write(json.dumps(DIc_))
        fw_.write("\n")
    
    fw_.close()
    print(f"✅ Semantic chunking completed! Output: relation_with_chucks_5_improved.json")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for relation, chucks in relation_chucks.items():
        print(f"{relation}: {len(chucks)} chunks")
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("SEMANTIC CHUNKING - Relation-Chunk Mapping")
    print("="*80)
    print("This script creates semantic chunks for each relation")
    print("Using: SemanticChunker (similarity-based boundaries)")
    print("Target: 5 chunks per relation (to match fixed-size)")
    print("="*80)
    print()
    
    target_chunks = 5  # Match fixed-size chunking
    make_chunck_semantic(target_chunks)
