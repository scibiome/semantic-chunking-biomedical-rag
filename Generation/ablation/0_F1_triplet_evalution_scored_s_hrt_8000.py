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


# Relations that are symmetric: A|R|B <=> B|R|A (same relation, swapped head/tail counts as correct)
SYMMETRIC_RELATIONS = {"interacts_with", "coexists_with", "associated_with"}

def parse_prediction_line(predicted_raw):
    """
    Parse prediction into (head, relation, tail). Normalizes malformed output:
    - Trailing | or comma: "h|r|t|" -> take first 3 parts
    - Extra tokens: "h|r|t|junk" -> take first 3 parts
    Returns (ph, pr, pt) or None if unparseable.
    """
    if not predicted_raw or not isinstance(predicted_raw, str):
        return None
    predicted_ = predicted_raw.strip().rstrip(',').rstrip('|').strip()
    parts = predicted_.split("|")
    if len(parts) >= 3:
        ph = normalize_entity(parts[0].lower().replace("       ", " ").strip())
        pr = parts[1].lower().strip()
        pt = normalize_entity(parts[2].lower().replace("       ", " ").strip())
        if ph != "" and pr != "" and pt != "":
            return (ph, pr, pt)
    return None

state_dict = {"p": 0, "c": 0, "g": 0}
def calclulate_f1(statics_dict, prefix=""):
    """
    Calculate the prec, recall and f1-score for the given state dict.
    """
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}


i=-1
# SCORED S-HRT 8000: Evaluate scored S-HRT model generation output (1-example)
with open("/home/users/riah/project_thesis/BIoMedRAG/4_generation_triple_model/test_triplet_chunk_scorer_scored_s_hrt_8000.json", "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        line=line.strip()
        line=json.loads(line)
        i=i+1
        P=set()
        sentence=line["context"].lower()
        gold_triples=set()
        ground_truth=line["ground_truth"]
       
        gold_t=ground_truth.split("|")
        if len(gold_t) != 3:
            continue
        # Normalize gold standard entities too (for fair comparison)
        gh=normalize_entity(gold_t[0].lower().replace("       "," ").strip())
        gr=gold_t[1].lower().strip()
        gt=normalize_entity(gold_t[2].lower().replace("       "," ").strip())
      
        gold_triples.add((gh,gr,gt))
            
        predictions=line["response"].split("\n\n")[0].split("\n")

        if len(predictions) >= 1:
            parsed = parse_prediction_line(predictions[0])
            if parsed is not None:
                ph, pr, pt = parsed
                P.add((ph, pr, pt))

        state_dict["p"] += len(P)
        state_dict["g"] += len(gold_triples)
        match = len(P & gold_triples)
        if match == 0 and P and gold_triples:
            (ph, pr, pt) = next(iter(P))
            (gh, gr, gt) = next(iter(gold_triples))
            if gr in SYMMETRIC_RELATIONS and (pt, pr, ph) == (gh, gr, gt):
                match = 1
        state_dict["c"] += match

all_metirc_results = calclulate_f1(state_dict, 'all')

print("="*80)
print("SCORED S-HRT MODEL (checkpoint 8000) EVALUATION RESULTS")
print("="*80)
print(all_metirc_results)
print(f"\nPrecision:  {all_metirc_results['all-prec']:.4f}")
print(f"Recall:     {all_metirc_results['all-recall']:.4f}")
print(f"F1-Score:   {all_metirc_results['all-f1']:.4f}")
print("="*80)

# Save results
with open("F1_EVALUATION_SCORED_S_HRT_8000.json", "w") as f:
    json.dump(all_metirc_results, f, indent=2)

with open("F1_EVALUATION_SCORED_S_HRT_8000.txt", "w") as f:
    f.write("SCORED S-HRT MODEL (checkpoint 8000) EVALUATION RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Precision:  {all_metirc_results['all-prec']:.4f}\n")
    f.write(f"Recall:     {all_metirc_results['all-recall']:.4f}\n")
    f.write(f"F1-Score:   {all_metirc_results['all-f1']:.4f}\n")
    f.write("="*80 + "\n")
    f.write(f"State dict: {state_dict}\n")

print("\nResults saved to F1_EVALUATION_SCORED_S_HRT_8000.json and F1_EVALUATION_SCORED_S_HRT_8000.txt")
