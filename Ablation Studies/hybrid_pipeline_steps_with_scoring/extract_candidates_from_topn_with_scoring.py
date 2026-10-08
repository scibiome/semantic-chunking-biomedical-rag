#!/usr/bin/env python3
"""
Extract all unique candidate chunks from Top-N selected chunks.
This creates a new candidate chunks file from what was actually selected,
rather than the entire pool.
"""

import json
import os
from collections import OrderedDict

# Input files (Top-N selections)
# Variant: Scoring pipeline - uses chunks from tiered trigger/hierarchy scoring
HYBRID_TOPN = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_train_data_topn_scored_5_with_scoring.json"
HYBRID_TOPN_TEST = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_test_data_topn_scored_5_with_scoring.json"

# Output files
OUTPUT_CANDIDATES = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_all_candiate_chunks_5_from_topn_with_scoring.txt"
OUTPUT_STATS = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_topn_candidates_stats_with_scoring.txt"

def extract_candidates_from_topn(topn_file, candidates_set, stats):
    """Extract all unique chunks from Top-N file."""
    print(f"\nProcessing: {os.path.basename(topn_file)}")
    
    total_examples = 0
    total_chunks = 0
    new_chunks = 0
    
    with open(topn_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                topn_chunks = data.get("topn_sim_chuck", {})
                
                if not topn_chunks:
                    continue
                
                total_examples += 1
                
                # Extract all chunks from this example
                for chunk_text, chunk_relation in topn_chunks.items():
                    total_chunks += 1
                    
                    # Use chunk text as key (deduplicate)
                    if chunk_text not in candidates_set:
                        candidates_set[chunk_text] = chunk_relation
                        new_chunks += 1
                    # Note: If same chunk appears with different relations, 
                    # we keep the first one encountered
                        
            except Exception as e:
                print(f"  Error at line {line_num}: {e}")
                continue
    
    stats[os.path.basename(topn_file)] = {
        'examples': total_examples,
        'total_chunks': total_chunks,
        'new_chunks': new_chunks
    }
    
    print(f"  Examples: {total_examples}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  New unique chunks: {new_chunks}")

def main():
    print("="*80)
    print("EXTRACTING CANDIDATE CHUNKS FROM TOP-N SELECTIONS")
    print("="*80)
    
    # Use OrderedDict to preserve insertion order (for reproducibility)
    candidates = OrderedDict()
    stats = {}
    
    # Process train Top-N
    if os.path.exists(HYBRID_TOPN):
        extract_candidates_from_topn(HYBRID_TOPN, candidates, stats)
    else:
        print(f"Warning: {HYBRID_TOPN} not found!")
    
    # Process test Top-N (if exists)
    if os.path.exists(HYBRID_TOPN_TEST):
        extract_candidates_from_topn(HYBRID_TOPN_TEST, candidates, stats)
    else:
        print(f"Warning: {HYBRID_TOPN_TEST} not found!")
    
    # Write candidate chunks file
    print(f"\n{'='*80}")
    print(f"Writing {len(candidates)} unique candidate chunks...")
    print(f"{'='*80}\n")
    
    with open(OUTPUT_CANDIDATES, 'w', encoding='utf-8') as f:
        for chunk_text in candidates.keys():
            f.write(f"{chunk_text}\n")
    
    print(f"✅ Written: {OUTPUT_CANDIDATES}")
    print(f"   Total unique chunks: {len(candidates)}")
    
    # Write statistics
    with open(OUTPUT_STATS, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CANDIDATE CHUNKS EXTRACTION STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Unique Candidate Chunks: {len(candidates)}\n\n")
        
        f.write("Source File Statistics:\n")
        f.write("-" * 80 + "\n")
        for filename, file_stats in stats.items():
            f.write(f"\n{filename}:\n")
            f.write(f"  Examples processed: {file_stats['examples']}\n")
            f.write(f"  Total chunks: {file_stats['total_chunks']}\n")
            f.write(f"  New unique chunks: {file_stats['new_chunks']}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NOTE: This file contains chunks that were ACTUALLY SELECTED\n")
        f.write("in the Top-N step, not the entire candidate pool.\n")
        f.write("This ensures the candidate space reflects high-quality chunks.\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Written: {OUTPUT_STATS}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total unique candidate chunks: {len(candidates)}")
    print(f"Output file: {OUTPUT_CANDIDATES}")
    print(f"Statistics file: {OUTPUT_STATS}")
    print(f"\n⚠️  NOTE: You'll need to generate embeddings for these chunks")
    print(f"   using hybrid_4_make_candidate_embeddings.py with this new file.")

if __name__ == "__main__":
    main()






"""
Extract all unique candidate chunks from Top-N selected chunks.
This creates a new candidate chunks file from what was actually selected,
rather than the entire pool.
"""

import json
import os
from collections import OrderedDict

# Input files (Top-N selections)
# Variant: Scoring pipeline - uses chunks from tiered trigger/hierarchy scoring
HYBRID_TOPN = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_train_data_topn_scored_5_with_scoring.json"
HYBRID_TOPN_TEST = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_test_data_topn_scored_5_with_scoring.json"

# Output files
OUTPUT_CANDIDATES = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_all_candiate_chunks_5_from_topn_with_scoring.txt"
OUTPUT_STATS = "/home/users/riah/project_thesis/BIoMedRAG/0_make_relation_chuck_and_scorer_data_IMPROVED/hybrid_pipeline_steps_with_scoring/hybrid_topn_candidates_stats_with_scoring.txt"

def extract_candidates_from_topn(topn_file, candidates_set, stats):
    """Extract all unique chunks from Top-N file."""
    print(f"\nProcessing: {os.path.basename(topn_file)}")
    
    total_examples = 0
    total_chunks = 0
    new_chunks = 0
    
    with open(topn_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                topn_chunks = data.get("topn_sim_chuck", {})
                
                if not topn_chunks:
                    continue
                
                total_examples += 1
                
                # Extract all chunks from this example
                for chunk_text, chunk_relation in topn_chunks.items():
                    total_chunks += 1
                    
                    # Use chunk text as key (deduplicate)
                    if chunk_text not in candidates_set:
                        candidates_set[chunk_text] = chunk_relation
                        new_chunks += 1
                    # Note: If same chunk appears with different relations, 
                    # we keep the first one encountered
                        
            except Exception as e:
                print(f"  Error at line {line_num}: {e}")
                continue
    
    stats[os.path.basename(topn_file)] = {
        'examples': total_examples,
        'total_chunks': total_chunks,
        'new_chunks': new_chunks
    }
    
    print(f"  Examples: {total_examples}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  New unique chunks: {new_chunks}")

def main():
    print("="*80)
    print("EXTRACTING CANDIDATE CHUNKS FROM TOP-N SELECTIONS")
    print("="*80)
    
    # Use OrderedDict to preserve insertion order (for reproducibility)
    candidates = OrderedDict()
    stats = {}
    
    # Process train Top-N
    if os.path.exists(HYBRID_TOPN):
        extract_candidates_from_topn(HYBRID_TOPN, candidates, stats)
    else:
        print(f"Warning: {HYBRID_TOPN} not found!")
    
    # Process test Top-N (if exists)
    if os.path.exists(HYBRID_TOPN_TEST):
        extract_candidates_from_topn(HYBRID_TOPN_TEST, candidates, stats)
    else:
        print(f"Warning: {HYBRID_TOPN_TEST} not found!")
    
    # Write candidate chunks file
    print(f"\n{'='*80}")
    print(f"Writing {len(candidates)} unique candidate chunks...")
    print(f"{'='*80}\n")
    
    with open(OUTPUT_CANDIDATES, 'w', encoding='utf-8') as f:
        for chunk_text in candidates.keys():
            f.write(f"{chunk_text}\n")
    
    print(f"✅ Written: {OUTPUT_CANDIDATES}")
    print(f"   Total unique chunks: {len(candidates)}")
    
    # Write statistics
    with open(OUTPUT_STATS, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CANDIDATE CHUNKS EXTRACTION STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Unique Candidate Chunks: {len(candidates)}\n\n")
        
        f.write("Source File Statistics:\n")
        f.write("-" * 80 + "\n")
        for filename, file_stats in stats.items():
            f.write(f"\n{filename}:\n")
            f.write(f"  Examples processed: {file_stats['examples']}\n")
            f.write(f"  Total chunks: {file_stats['total_chunks']}\n")
            f.write(f"  New unique chunks: {file_stats['new_chunks']}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NOTE: This file contains chunks that were ACTUALLY SELECTED\n")
        f.write("in the Top-N step, not the entire candidate pool.\n")
        f.write("This ensures the candidate space reflects high-quality chunks.\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Written: {OUTPUT_STATS}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total unique candidate chunks: {len(candidates)}")
    print(f"Output file: {OUTPUT_CANDIDATES}")
    print(f"Statistics file: {OUTPUT_STATS}")
    print(f"\n⚠️  NOTE: You'll need to generate embeddings for these chunks")
    print(f"   using hybrid_4_make_candidate_embeddings.py with this new file.")

if __name__ == "__main__":
    main()











