#!/usr/bin/env python3
"""Check if training files contain chunk examples in instructions"""
import json

def check_chunks_in_file(filename):
    print(f"\n📋 Checking: {filename}")
    print("=" * 80)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    samples_with_chunks = 0
    samples_without_chunks = 0
    
    # Check first few samples in detail
    print("\n🔍 First 3 samples:")
    for i in range(min(3, len(lines))):
        obj = json.loads(lines[i])
        instruction = obj['instruction']
        
        # Count "Context:" occurrences (each chunk example has Context: and Response:)
        context_count = instruction.count("Context:")
        response_count = instruction.count("Response:")
        
        print(f"\n  Sample {i+1}:")
        print(f"    Response: {obj['response']}")
        print(f"    Instruction length: {len(instruction)} chars")
        print(f"    'Context:' occurrences: {context_count}")
        print(f"    'Response:' occurrences: {response_count}")
        
        if context_count > 2:  # Base instruction has 2 examples, chunks add more
            print(f"    ✅ HAS CHUNKS (likely {context_count - 2} chunk examples)")
            samples_with_chunks += 1
        else:
            print(f"    ⚠️  NO CHUNKS (only base examples)")
            samples_without_chunks += 1
        
        # Show a snippet of instruction
        if "Examples:" in instruction:
            examples_part = instruction.split("Examples:")[1][:200]
            print(f"    Examples snippet: {examples_part}...")
    
    # Check all samples
    print(f"\n📊 Overall statistics:")
    for line in lines:
        obj = json.loads(line)
        instruction = obj['instruction']
        context_count = instruction.count("Context:")
        
        if context_count > 2:
            samples_with_chunks += 1
        else:
            samples_without_chunks += 1
    
    print(f"    Total samples: {total_samples}")
    print(f"    Samples WITH chunks: {samples_with_chunks} ({samples_with_chunks/total_samples*100:.1f}%)")
    print(f"    Samples WITHOUT chunks: {samples_without_chunks} ({samples_without_chunks/total_samples*100:.1f}%)")
    
    if samples_with_chunks > 0:
        print(f"\n    ✅ File contains chunk examples!")
    else:
        print(f"\n    ❌ File does NOT contain chunk examples!")
    
    return samples_with_chunks, samples_without_chunks

print("=" * 80)
print("DDI FIXED - Chunk Examples Verification")
print("=" * 80)

# Check train file
train_with, train_without = check_chunks_in_file("train_chuck_5_triplet_ddi_fixed.json")

# Check test file
test_with, test_without = check_chunks_in_file("test_chuck_5_triplet_ddi_fixed.json")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Train: {train_with} with chunks, {train_without} without")
print(f"Test:  {test_with} with chunks, {test_without} without")

if train_with > 0 and test_with > 0:
    print("\n✅ BOTH files contain chunk examples!")
else:
    print("\n⚠️  One or both files are missing chunk examples")
print("=" * 80)
