#!/usr/bin/env python3
"""Verify generated training data files"""
import json
import os

print("=" * 80)
print("DDI FIXED - Training Data Verification")
print("=" * 80)
print()

# Check files
train_file = "train_chuck_5_triplet_ddi_fixed.json"
test_file = "test_chuck_5_triplet_ddi_fixed.json"

for filename, expected_count in [(train_file, 1027), (test_file, 1094)]:
    if not os.path.exists(filename):
        print(f"❌ Missing: {filename}")
        continue
    
    print(f"📊 {filename}")
    print("-" * 80)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        count = len(lines)
        
    print(f"  Total samples: {count}")
    print(f"  Expected: ~{expected_count}")
    
    if count != expected_count:
        print(f"  ⚠️  Sample count mismatch!")
    else:
        print(f"  ✓ Sample count matches")
    
    # Check structure
    sample = json.loads(lines[0])
    print(f"  Keys: {list(sample.keys())}")
    
    required_keys = ['instruction', 'context', 'response', 'category']
    missing = [k for k in required_keys if k not in sample]
    if missing:
        print(f"  ❌ Missing keys: {missing}")
    else:
        print(f"  ✓ All required keys present")
    
    # Check content
    print(f"\n  Sample content:")
    print(f"    Response: {sample['response']}")
    print(f"    Context (first 80 chars): {sample['context'][:80]}...")
    print(f"    Instruction length: {len(sample['instruction'])} chars")
    print(f"    Category: {sample.get('category', 'N/A')}")
    
    # Count relations
    relations = {}
    for line in lines:
        obj = json.loads(line)
        response = obj['response']
        # Extract relation
        parts = response.split('|')
        if len(parts) == 3:
            rel = parts[1]
            relations[rel] = relations.get(rel, 0) + 1
    
    print(f"\n  Relation distribution:")
    for rel, cnt in sorted(relations.items(), key=lambda x: -x[1]):
        pct = (cnt / count) * 100
        print(f"    {rel:12s}: {cnt:4d} ({pct:5.1f}%)")
    
    print()

print("=" * 80)
print("✅ Verification complete!")
print("=" * 80)
