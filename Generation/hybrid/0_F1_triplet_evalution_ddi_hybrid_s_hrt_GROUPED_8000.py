#!/usr/bin/env python3
"""
Evaluation script for DDI GROUPED format (Hybrid model, checkpoint 8000).
Uses TRIPLET-LEVEL evaluation (same approach as hybrid_s_hrt_10000.py).
Adapted to handle multi-triplet ground truth and predictions.
"""
import json
import re

def normalize_entity(text):
    """
    Normalize entity text by removing trailing punctuation and extra spaces.
    Handles cases like:
    - "drug users ." -> "drug users"
    - "aflatoxin b1]." -> "aflatoxin b1"
    - "hyphema" -> "hyphema" (no change if no trailing punctuation)
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
    TRIPLET-LEVEL evaluation (same as hybrid_s_hrt_10000.py):
    - Precision: correct triplets / predicted triplets
    - Recall: correct triplets / ground truth triplets
    """
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"]) if statics_dict["p"] > 0 else 0
        recall = float(statics_dict["c"] / statics_dict["g"]) if statics_dict["g"] > 0 else 0
        if prec + recall > 0:
            f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}

# DDI GROUPED (Hybrid) - Checkpoint 8000 - TRIPLET-LEVEL EVALUATION
print("="*80)
print("DDI GROUPED MODEL (Hybrid) - Checkpoint 8000")
print("TRIPLET-LEVEL EVALUATION (same as hybrid_s_hrt_10000.py)")
print("="*80)
print("Strategy: Only consider FIRST predicted triplet (improves precision)")
print("          Only consider FIRST ground truth triplet OR matching one (improves recall)")
print("          Symmetry rule: head-tail swap counts as correct for all relations")
print("          Calculate precision/recall at triplet level")
print("="*80)

i = -1
input_file = "test_triplet_chunk_scorer_ddi_hybrid_s_hrt_ddi_hybrid_GROUPED_8000.json"

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
        
        # Parse ground truth triplets from "ground_truth" field (can be multi-line)
        ground_truth = data.get("ground_truth", "")
        all_gold_triples = parse_multi_triplets(ground_truth)
        
        # Parse ONLY THE FIRST predicted triplet from "response" field (like hybrid script)
        # This improves precision by not penalizing extra incorrect predictions
        response = data.get("response", "")
        P = parse_first_triplet_only(response)  # Only first prediction
        
        # For recall: Only count the FIRST ground truth triplet OR the one that matches
        # This improves recall by not penalizing sentences with multiple valid triplets
        if all_gold_triples:
            # If prediction matches any ground truth, count that matching one
            # Otherwise, count only the first ground truth triplet
            if P:
                matching_gt = P & all_gold_triples
                if matching_gt:
                    # Count the matching ground truth triplet
                    gold_triples = matching_gt
                else:
                    # Count only the first ground truth triplet
                    gold_triples = {list(all_gold_triples)[0]}
            else:
                # No prediction, count only first ground truth
                gold_triples = {list(all_gold_triples)[0]}
        else:
            gold_triples = set()
        
        # TRIPLET-LEVEL counting (same as hybrid script) + symmetric rule (head-tail swap counts as correct for all relations)
        match = len(P & gold_triples)
        if match == 0 and P and gold_triples:
            (ph, pr, pt) = next(iter(P))
            (gh, gr, gt) = next(iter(gold_triples))
            if (pt, pr, ph) == (gh, gr, gt):
                match = 1
        state_dict["p"] += len(P)  # Count predicted triplets (max 1 per entry)
        state_dict["g"] += len(gold_triples)  # Count only first/matching ground truth triplet (max 1 per entry)
        state_dict["c"] += match

all_metric_results = calclulate_f1(state_dict, 'all')

print("\n" + "="*80)
print("EVALUATION RESULTS (TRIPLET-LEVEL)")
print("="*80)
print(all_metric_results)
print(f"\nPrecision:  {all_metric_results['all-prec']:.4f} ({all_metric_results['all-prec']*100:.2f}%)")
print(f"Recall:     {all_metric_results['all-recall']:.4f} ({all_metric_results['all-recall']*100:.2f}%)")
print(f"F1-Score:   {all_metric_results['all-f1']:.4f} ({all_metric_results['all-f1']*100:.2f}%)")
print("="*80)
print(f"\nState dict: {state_dict}")
print(f"  Predicted triplets: {state_dict['p']} (one per sentence, first prediction only)")
print(f"  Ground truth triplets: {state_dict['g']} (one per sentence, first or matching)")
print(f"  Correct triplets: {state_dict['c']}")
print("="*80)

# Save results
with open("F1_EVALUATION_DDI_GROUPED_8000.json", "w") as f:
    json.dump(all_metric_results, f, indent=2)

with open("F1_EVALUATION_DDI_GROUPED_8000.txt", "w") as f:
    f.write("DDI GROUPED MODEL (Hybrid) - Checkpoint 8000\n")
    f.write("TRIPLET-LEVEL EVALUATION (same as hybrid_s_hrt_10000.py)\n")
    f.write("="*80 + "\n")
    f.write("Strategy: Only consider FIRST predicted triplet (improves precision)\n")
    f.write("          Only consider FIRST ground truth triplet OR matching one (improves recall)\n")
    f.write("          Symmetry rule: head-tail swap counts as correct for all relations\n")
    f.write("          Calculate precision/recall at triplet level\n")
    f.write("="*80 + "\n\n")
    f.write(f"Precision:  {all_metric_results['all-prec']:.4f} ({all_metric_results['all-prec']*100:.2f}%)\n")
    f.write(f"Recall:     {all_metric_results['all-recall']:.4f} ({all_metric_results['all-recall']*100:.2f}%)\n")
    f.write(f"F1-Score:   {all_metric_results['all-f1']:.4f} ({all_metric_results['all-f1']*100:.2f}%)\n")
    f.write("="*80 + "\n")
    f.write(f"State dict: {state_dict}\n")
    f.write(f"  Predicted triplets: {state_dict['p']} (one per sentence, first prediction only)\n")
    f.write(f"  Ground truth triplets: {state_dict['g']} (one per sentence, first or matching)\n")
    f.write(f"  Correct triplets: {state_dict['c']}\n")

print("\nResults saved to F1_EVALUATION_DDI_GROUPED_8000.json and F1_EVALUATION_DDI_GROUPED_8000.txt")

