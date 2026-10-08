import json
from transformers import LlamaTokenizer

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained("/home/users/riah/models/Llama-2-13b-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Training file
train_file = "train_triplet_chunk_scorer_improved_filtered.json"
test_file = "test_triplet_chunk_scorer_improved_filtered.json"

def format_example(example):
    """Format example same way as trainer"""
    if example.get("context", "") != "":
        input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Input: \n"
        f"{example['context']}\n\n"
        f"### Response: \n"
        f"{example['response']}")
    else:
        input_prompt = (f"Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}")
    return input_prompt

def check_token_lengths(file_path, file_type="Train"):
    print(f"\n{'='*80}")
    print(f"Checking {file_type} File: {file_path}")
    print(f"{'='*80}")
    
    token_lengths = []
    truncated_count = 0
    max_seq_length = 512
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line.strip())
                formatted_text = format_example(example)
                tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
                token_length = len(tokens)
                token_lengths.append(token_length)
                
                if token_length > max_seq_length:
                    truncated_count += 1
                    
                if i < 5:  # Show first 5 examples
                    print(f"\nExample {i+1}:")
                    print(f"  Token length: {token_length}")
                    print(f"  Truncated (>512): {'Yes' if token_length > max_seq_length else 'No'}")
                    if token_length > max_seq_length:
                        print(f"  Would be truncated by: {token_length - max_seq_length} tokens")
                    
            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                continue
    
    if token_lengths:
        print(f"\n{'='*80}")
        print(f"{file_type} Token Length Statistics:")
        print(f"{'='*80}")
        print(f"Total examples: {len(token_lengths)}")
        print(f"Min tokens: {min(token_lengths)}")
        print(f"Max tokens: {max(token_lengths)}")
        print(f"Mean tokens: {sum(token_lengths) / len(token_lengths):.2f}")
        print(f"Median tokens: {sorted(token_lengths)[len(token_lengths)//2]}")
        print(f"\nTruncation Analysis (max_seq_length={max_seq_length}):")
        print(f"  Truncated examples: {truncated_count} ({100*truncated_count/len(token_lengths):.2f}%)")
        print(f"  Not truncated: {len(token_lengths) - truncated_count} ({100*(len(token_lengths)-truncated_count)/len(token_lengths):.2f}%)")
        
        # Show distribution
        bins = [0, 256, 384, 512, 768, 1024, float('inf')]
        bin_labels = ['0-256', '257-384', '385-512', '513-768', '769-1024', '1024+']
        print(f"\nToken Length Distribution:")
        for i, (bin_min, bin_max) in enumerate(zip(bins[:-1], bins[1:])):
            count = sum(1 for tl in token_lengths if bin_min < tl <= bin_max)
            print(f"  {bin_labels[i]}: {count} ({100*count/len(token_lengths):.2f}%)")
    
    return token_lengths

# Check train file
train_lengths = check_token_lengths(train_file, "Train")

# Check test file
test_lengths = check_token_lengths(test_file, "Test")

print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}")
print(f"Train: {len(train_lengths)} examples")
print(f"Test: {len(test_lengths)} examples")
print(f"\nRecommendation:")
if max(train_lengths + test_lengths) > 512:
    print(f"  ⚠️  Some examples exceed 512 tokens. Consider:")
    print(f"     • Increasing max_seq_length to 768 or 1024")
    print(f"     • Or accept {100*sum(1 for tl in train_lengths + test_lengths if tl > 512)/len(train_lengths + test_lengths):.2f}% truncation")
else:
    print(f"  ✅ All examples fit within 512 tokens!")


