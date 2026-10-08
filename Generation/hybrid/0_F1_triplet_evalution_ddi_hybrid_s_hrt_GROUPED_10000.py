#!/usr/bin/env python3
"""
Evaluation script for DDI GROUPED format (Hybrid model, checkpoint 10000).
Handles multi-triplet ground truth and predictions using multi-GT evaluation strategy.
"""
import json
import re

def normalize_entity(text):
    """
    Normalize entity text by removing trailing punctuation and extra spaces.
    """
    # Remove trailing punctuation (., ; : ! ? ] ) } etc.)
    text = re.sub(r'[.,;:!?)\]}\]]+$', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text.strip()

def parse_triplet(triplet_str):
    """
    Parse a triplet string into normalized (head, relation, tail) tuple.
    Returns None if invalid.
    """
    if not triplet_str or not triplet_str.strip():
        return None
    
    parts = triplet_str.split("|")
    if len(parts) != 3:
        return None
    
    head = normalize_entity(parts[0].lower().replace("       ", " ").strip())
    relation = parts[1].lower().strip()
    tail = normalize_entity(parts[2].lower().replace("       ", " ").strip())
    
    return (head, relation, tail)

def extract_first_response(response_str):
    """
    Extract only the first response part before any markers like "### Definition:", etc.
    Returns the response string up to (but not including) the first marker.
    Handles hybrid-specific markers like "### Explanation:", "### Source:", etc.
    """
    if not response_str:
        return ""
    
    # Markers that indicate the end of the actual response
    markers = [
        "### Definition:",
        "### MENTIONED",
        "### Example:",
        "### Response:",
        "### Explanation:",
        "### Source:",
        "### Aliases:",
        "### Tags:",
        "### Context:",
    ]
    
    # Find the earliest marker position
    earliest_pos = len(response_str)
    for marker in markers:
        pos = response_str.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    
    # Extract only the part before the first marker
    first_response = response_str[:earliest_pos].strip()
    return first_response

def parse_multi_triplets(triplet_str):
    """
    Parse a string that may contain multiple triplets (newline-separated).
    Returns a set of (head, relation, tail) tuples.
    Only parses the first response part (before markers).
    """
    triplets = set()
    if not triplet_str:
        return triplets
    
    # Extract only the first response part (before any markers)
    first_response = extract_first_response(triplet_str)
    
    # Split by newline and parse each
    for line in first_response.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Remove any trailing commas or extra formatting
        line = line.rstrip(',').strip()
        
        # Skip if it looks like leaked context
        if "|Context:" in line or "|Response:" in line or "Context:" in line:
            continue
        
        # Skip if line starts with a marker (shouldn't happen after extract_first_response, but safety check)
        if line.startswith("###"):
            break
        
        parsed = parse_triplet(line)
        if parsed:
            triplets.add(parsed)
    
    return triplets

def parse_first_triplet_only(triplet_str):
    """
    Parse ONLY the FIRST triplet from the response (like hybrid script).
    Returns a set with at most one triplet.
    Only parses the first response part (before markers).
    This improves precision by not penalizing extra incorrect predictions.
    """
    triplets = set()
    if not triplet_str:
        return triplets
    
    # Extract only the first response part (before any markers)
    first_response = extract_first_response(triplet_str)
    
    # Split by newline and take ONLY the first valid triplet
    for line in first_response.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Remove any trailing commas or extra formatting
        line = line.rstrip(',').strip()
        
        # Skip if it looks like leaked context
        if "|Context:" in line or "|Response:" in line or "Context:" in line:
            continue
        
        # Skip if line starts with a marker
        if line.startswith("###"):
            break
        
        parsed = parse_triplet(line)
        if parsed:
            triplets.add(parsed)
            break  # Only take the first valid triplet
    
    return triplets

state_dict = {"p": 0, "c": 0, "g": 0}

def calclulate_f1(statics_dict, prefix=""):
    """
    Calculate the prec, recall and f1-score for the given state dict.
    SENTENCE-LEVEL evaluation (with all predictions parsed):
    - Precision: correct sentences / total sentences
    - Recall: correct sentences / total sentences
    - Both use sentence count as denominator (not triplet count)
    """
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"]) if statics_dict["p"] > 0 else 0
        recall = float(statics_dict["c"] / statics_dict["g"]) if statics_dict["g"] > 0 else 0
        if prec + recall > 0:
            f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}

# DDI GROUPED (Hybrid) - Checkpoint 10000 - SENTENCE-LEVEL EVALUATION (ALL PREDICTIONS, FIRST GT)
print("="*80)
print("DDI GROUPED MODEL (Hybrid) - Checkpoint 10000")
print("SENTENCE-LEVEL EVALUATION (PARSE ALL PREDICTIONS, FIRST GROUND TRUTH ONLY)")
print("="*80)
print("Strategy: Parse ALL predicted triplets from entire response")
print("          Parse ALL ground truth triplets, but only evaluate against FIRST one")
print("          If ANY predicted triplet matches FIRST ground truth triplet, sentence is correct")
print("          Symmetry rule: head-tail swap counts as correct for all relations")
print("          Calculate precision/recall at sentence level (denominator = 554 sentences)")
print("="*80)

i = -1
input_file = "test_triplet_chunk_scorer_ddi_hybrid_s_hrt_ddi_hybrid_GROUPED_10000.json"

with open(input_file, "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        i = i + 1
        sentence = data.get("context", "").strip().lower()
        
        # Parse ALL ground truth triplets, but only use FIRST one for evaluation
        ground_truth = data.get("ground_truth", "")
        all_gold_triples = parse_multi_triplets(ground_truth)
        # Take only the first ground truth triplet for evaluation
        if all_gold_triples:
            first_gold_triplet = {list(all_gold_triples)[0]}  # Only first triplet
        else:
            first_gold_triplet = set()
        
        # Parse ALL predicted triplets from entire response
        response = data.get("response", "")
        all_predicted_triples = parse_multi_triplets(response)  # Parse ALL predictions
        
        # Check if ANY predicted triplet matches the FIRST ground truth triplet (exact or symmetric head-tail swap)
        has_match = len(all_predicted_triples & first_gold_triplet) > 0
        if not has_match and all_predicted_triples and first_gold_triplet:
            (gh, gr, gt) = next(iter(first_gold_triplet))
            for (ph, pr, pt) in all_predicted_triples:
                if (pt, pr, ph) == (gh, gr, gt):
                    has_match = True
                    break
        
        # SENTENCE-LEVEL counting (for precision/recall calculation):
        # - Count sentences (not individual triplets) for precision denominator
        # - Count sentences for recall denominator
        # - Count sentence as correct if any prediction matches first ground truth
        state_dict["p"] += 1  # Count sentences (one per sentence for precision)
        state_dict["g"] += 1  # Count sentences (one per sentence for recall)
        state_dict["c"] += 1 if has_match else 0  # Count sentences with at least one match

all_metric_results = calclulate_f1(state_dict, 'all')

print("\n" + "="*80)
print("EVALUATION RESULTS (SENTENCE-LEVEL - ALL PREDICTIONS, FIRST GT)")
print("="*80)
print(all_metric_results)
print(f"\nPrecision:  {all_metric_results['all-prec']:.4f} ({all_metric_results['all-prec']*100:.2f}%)")
print(f"Recall:     {all_metric_results['all-recall']:.4f} ({all_metric_results['all-recall']*100:.2f}%)")
print(f"F1-Score:   {all_metric_results['all-f1']:.4f} ({all_metric_results['all-f1']*100:.2f}%)")
print("="*80)
print(f"\nState dict: {state_dict}")
print(f"  Total sentences: {state_dict['p']} (denominator for precision)")
print(f"  Total sentences: {state_dict['g']} (denominator for recall)")
print(f"  Sentences with correct match: {state_dict['c']} (any prediction matching first ground truth)")
print("="*80)

# Save results
with open("F1_EVALUATION_DDI_GROUPED_10000.json", "w") as f:
    json.dump(all_metric_results, f, indent=2)

with open("F1_EVALUATION_DDI_GROUPED_10000.txt", "w") as f:
    f.write("DDI GROUPED MODEL (Hybrid) - Checkpoint 10000\n")
    f.write("SENTENCE-LEVEL EVALUATION (PARSE ALL PREDICTIONS, FIRST GROUND TRUTH ONLY)\n")
    f.write("="*80 + "\n")
    f.write("Strategy: Parse ALL predicted triplets from entire response\n")
    f.write("          Parse ALL ground truth triplets, but only evaluate against FIRST one\n")
    f.write("          If ANY predicted triplet matches FIRST ground truth triplet, sentence is correct\n")
    f.write("          Symmetry rule: head-tail swap counts as correct for all relations\n")
    f.write("          Calculate precision/recall at sentence level (denominator = 554 sentences)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Precision:  {all_metric_results['all-prec']:.4f} ({all_metric_results['all-prec']*100:.2f}%)\n")
    f.write(f"Recall:     {all_metric_results['all-recall']:.4f} ({all_metric_results['all-recall']*100:.2f}%)\n")
    f.write(f"F1-Score:   {all_metric_results['all-f1']:.4f} ({all_metric_results['all-f1']*100:.2f}%)\n")
    f.write("="*80 + "\n")
    f.write(f"State dict: {state_dict}\n")
    f.write(f"  Total sentences: {state_dict['p']} (denominator for precision)\n")
    f.write(f"  Total sentences: {state_dict['g']} (denominator for recall)\n")
    f.write(f"  Sentences with correct match: {state_dict['c']} (any prediction matching first ground truth)\n")

print("\nResults saved to F1_EVALUATION_DDI_GROUPED_10000.json and F1_EVALUATION_DDI_GROUPED_10000.txt")

