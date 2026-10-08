#!/usr/bin/env python3
"""
Script to merge all shard files into a single numpy array file.
This preserves the original data structure and doesn't disturb the pipeline.
"""

import numpy as np
import glob
import os
from pathlib import Path

def merge_shard_files():
    """Merge all shard files into a single numpy array."""
    
    # Get all shard files in sorted order
    shard_pattern = "all_posible_chuck_vector_5_medllama_13b_shard_*.npy"
    shard_files = sorted(glob.glob(shard_pattern))
    
    print(f"Found {len(shard_files)} shard files to merge")
    
    if not shard_files:
        print("No shard files found!")
        return
    
    # Load and concatenate all shard files
    print("Loading shard files...")
    arrays = []
    
    for i, shard_file in enumerate(shard_files):
        if i % 100 == 0:
            print(f"Processing shard {i+1}/{len(shard_files)}: {shard_file}")
        
        try:
            shard_data = np.load(shard_file)
            arrays.append(shard_data)
        except Exception as e:
            print(f"Error loading {shard_file}: {e}")
            continue
    
    if not arrays:
        print("No valid shard files loaded!")
        return
    
    print("Concatenating arrays...")
    # Concatenate all arrays along the first axis
    merged_array = np.concatenate(arrays, axis=0)
    
    print(f"Merged array shape: {merged_array.shape}")
    print(f"Merged array dtype: {merged_array.dtype}")
    
    # Save the merged array
    output_file = "all_posible_chuck_vector_5_medllama_13b_merged.npy"
    print(f"Saving merged array to {output_file}...")
    np.save(output_file, merged_array)
    
    # Verify the saved file
    print("Verifying saved file...")
    loaded_array = np.load(output_file)
    print(f"Verification - Loaded array shape: {loaded_array.shape}")
    print(f"Verification - Arrays match: {np.array_equal(merged_array, loaded_array)}")
    
    # Calculate total size
    total_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"Merged file size: {total_size:.2f} MB")
    
    print("✅ Merge completed successfully!")
    print(f"Original shard files: {len(shard_files)}")
    print(f"Merged file: {output_file}")

if __name__ == "__main__":
    merge_shard_files()

