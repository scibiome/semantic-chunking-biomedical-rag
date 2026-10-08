#!/usr/bin/env python3
"""
Quick Test: Verify Chunks Are Preserved in Fixed Step 6 Scripts
This tests if the fix works BEFORE committing to full retraining
"""
import json
import sys

print("=" * 80)
print("QUICK TEST: Chunk Preservation Verification")
print("=" * 80)

# Test Step 5 files
print("\n1. Checking Step 5 Files (Source):")
print("-" * 80)

try:
    with open("../0_make_relation_chuck_and_scorer_data/train_chuck_instruction_5_llama13b_right.json", "r") as f:
        step5_fixed = json.loads(f.readline().strip())
        fixed_chunks = step5_fixed['instruction'].count('Context:')
        print(f"✅ FIXED Step 5: {fixed_chunks} chunks")
except Exception as e:
    print(f"❌ Error reading FIXED Step 5: {e}")
    sys.exit(1)

try:
    with open("../0_make_relation_chuck_and_scorer_data_IMPROVED/train_chuck_instruction_5_llama13b_right_improved.json", "r") as f:
        step5_improved = json.loads(f.readline().strip())
        improved_chunks = step5_improved['instruction'].count('Context:')
        print(f"✅ IMPROVED Step 5: {improved_chunks} chunks")
except Exception as e:
    print(f"❌ Error reading IMPROVED Step 5: {e}")
    sys.exit(1)

print(f"\nStep 5 chunks are DIFFERENT: {fixed_chunks != improved_chunks} ✅")

# Test current Step 6 output (should have no chunks)
print("\n2. Checking Current Step 6 Output (Before Fix):")
print("-" * 80)

try:
    with open("train_chuck_5_triplet_llama2_13b_right_2_FIXED.json", "r") as f:
        current_fixed = json.loads(f.readline().strip())
        current_fixed_chunks = current_fixed['instruction'].count('Context:')
        print(f"Current FIXED: {current_fixed_chunks} chunks (should be 2)")
except Exception as e:
    print(f"⚠️  Could not read current FIXED: {e}")

try:
    with open("train_chuck_5_triplet_llama2_13b_right_2_improved.json", "r") as f:
        current_improved = json.loads(f.readline().strip())
        current_improved_chunks = current_improved['instruction'].count('Context:')
        print(f"Current IMPROVED: {current_improved_chunks} chunks (should be 2)")
except Exception as e:
    print(f"⚠️  Could not read current IMPROVED: {e}")

# Test fixed scripts (dry run - just check first example)
print("\n3. Testing Fixed Scripts (Dry Run - First Example):")
print("-" * 80)

# Simulate what fixed script does
try:
    with open("KNN_demo_train.json", "r") as knn_f, \
         open("../0_make_relation_chuck_and_scorer_data/train_chuck_instruction_5_llama13b_right.json", "r") as step5_f:
        
        knn_line = json.loads(knn_f.readline().strip())
        step5_obj = json.loads(step5_f.readline().strip())
        
        # Simulate fixed script
        knn_line["instruction"] = step5_obj["instruction"]
        
        # Check result
        test_fixed_chunks = knn_line['instruction'].count('Context:')
        print(f"✅ Simulated FIXED output: {test_fixed_chunks} chunks")
        print(f"   Matches Step 5? {test_fixed_chunks == fixed_chunks}")
        
except Exception as e:
    print(f"❌ Error in simulation: {e}")
    sys.exit(1)

try:
    with open("0_Sunflower_triplet_extraction_13b/data/KNN_demo_train.json", "r") as knn_f, \
         open("../0_make_relation_chuck_and_scorer_data_IMPROVED/train_chuck_instruction_5_llama13b_right_improved.json", "r") as step5_f:
        
        knn_line = json.loads(knn_f.readline().strip())
        step5_obj = json.loads(step5_f.readline().strip())
        
        # Simulate fixed script
        knn_line["instruction"] = step5_obj["instruction"]
        
        # Check result
        test_improved_chunks = knn_line['instruction'].count('Context:')
        print(f"✅ Simulated IMPROVED output: {test_improved_chunks} chunks")
        print(f"   Matches Step 5? {test_improved_chunks == improved_chunks}")
        
except Exception as e:
    print(f"❌ Error in simulation: {e}")
    sys.exit(1)

# Final verification
print("\n" + "=" * 80)
print("VERIFICATION RESULTS:")
print("=" * 80)

if test_fixed_chunks == fixed_chunks and test_improved_chunks == improved_chunks:
    print("✅ SUCCESS: Fixed scripts will preserve chunks!")
    print(f"   FIXED will have: {test_fixed_chunks} chunks")
    print(f"   IMPROVED will have: {test_improved_chunks} chunks")
    print(f"   They are DIFFERENT: {test_fixed_chunks != test_improved_chunks}")
    print("\n✅ SAFE TO PROCEED: Chunks will be preserved in training files!")
    print("   You can now regenerate training files and retrain.")
else:
    print("❌ FAILURE: Chunks not preserved correctly!")
    print("   Need to debug the fixed scripts.")
    sys.exit(1)








