import json

## Note: Model loading not needed here; we only reorganize precomputed similar chunks from Step 2

def make_topn_for_train():
    """Extract top-N similar chunks from Step 2 train output"""
    fw_ = open("train_data_topn_chucks_5_improved.json", "w")
    
    relation_chucks = {}
    j = -1
    all_l = []
    
    print("Processing training data from Step 2 output...")
    with open("train_data_chucks_5_improved.json", "r") as fr:
        for line in fr.readlines():
            j = j + 1
            if j % 100 == 0:
                print(f"Processed {j} train samples...")
            
            line = json.loads(line)
            
            all_chucks = line["all_chucks"]  # Already computed in Step 2
            
            # Extract top-1 from each chunk's top-5 similar chunks
            all_c_top_n = {}
            for chucks in all_chucks:
                for ch, itssimchuk in chucks.items():
                    # itssimchuk[0] = top-5 chunks, itssimchuk[1] = their relations
                    # We take only the top-1 (index 0)
                    for i in range(len(itssimchuk[0][:1])):
                        all_c_top_n[itssimchuk[0][i]] = itssimchuk[1][i]
            
            line["topn_sim_chuck"] = all_c_top_n
            fw_.write(json.dumps(line))
            fw_.flush()
            fw_.write("\n")
    
    fw_.close()
    print(f"✓ Training topn complete: {j+1} samples processed")


def make_topn_for_test():
    """Extract top-N similar chunks from Step 2 test output (deduplicated)"""
    fw_ = open("test_data_topn_chucks_5_improved_deduplicated.json", "w")
    
    relation_chucks = {}
    j = -1
    all_l = []
    
    print("\nProcessing test data from Step 2 output (deduplicated)...")
    with open("test_data_chucks_5_improved_deduplicated.json", "r") as fr:
        for line in fr.readlines():
            j = j + 1
            if j % 100 == 0:
                print(f"Processed {j} test samples...")
            
            line = json.loads(line)
            
            all_chucks = line["all_chucks"]  # Already computed in Step 2
            
            # Extract top-1 from each chunk's top-5 similar chunks
            all_c_top_n = {}
            for chucks in all_chucks:
                for ch, itssimchuk in chucks.items():
                    # itssimchuk[0] = top-5 chunks, itssimchuk[1] = their relations
                    # We take only the top-1 (index 0)
                    for i in range(len(itssimchuk[0][:1])):
                        all_c_top_n[itssimchuk[0][i]] = itssimchuk[1][i]
            
            line["topn_sim_chuck"] = all_c_top_n
            fw_.write(json.dumps(line))
            fw_.flush()
            fw_.write("\n")
    
    fw_.close()
    print(f"✓ Test topn complete: {j+1} samples processed")


if __name__ == "__main__":
    print("="*80)
    print("Step 3: Extract TopN Similar Chunks (Semantic Pipeline)")
    print("="*80)
    print("\nThis step extracts top-N similar chunks from Step 2's output")
    print("No model loading or chunking - just data reorganization\n")
    
    make_topn_for_train()
    make_topn_for_test()
    
    print("\n" + "="*80)
    print("✅ Step 3 Complete!")
    print("="*80)
    print("\nOutput files:")
    print("  - train_data_topn_chucks_5_improved.json")
    print("  - test_data_topn_chucks_5_improved_deduplicated.json")
    print("\nNext: Run Step 4 (candidate chunk embeddings)")
    print("="*80)
