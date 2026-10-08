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

def extract_first_response(response_str):
    """
    Extract the first valid triplet from the response, ignoring extra sections.
    """
    if not response_str:
        return ""
    
    # Split by newlines and take the first non-empty line
    lines = response_str.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that look like section headers
        if line.startswith('###') or line.startswith('##') or line.startswith('#'):
            continue
        # Check if it looks like a triplet (contains |)
        if '|' in line:
            # Take only the part before any additional markers
            if '###' in line:
                line = line.split('###')[0].strip()
            return line
    return ""

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
# CHEMPROT FIXED S-HRT 10000: Evaluate ChemProt fixed S-HRT model generation output
with open("/home/users/riah/project_thesis/BIoMedRAG/4_generation_triple_model/test_triplet_chunk_scorer_chemprot_fixed_s_hrt_10000.json", "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        line=line.strip()
        if not line:
            continue
        line=json.loads(line)
        i=i+1
        P=set()
        sentence=line["context"].lower()
        gold_triples=set()
        ground_truth=line["ground_truth"]
       
        gold_t=ground_truth.split("|")
        if len(gold_t) != 3:
            continue
        # Normalize gold standard entities
        gh=normalize_entity(gold_t[0].lower().replace("       "," ").strip())
        gr=gold_t[1].lower().strip()
        gt=normalize_entity(gold_t[2].lower().replace("       "," ").strip())
      
        gold_triples.add((gh,gr,gt))
            
        # Extract first response from model output
        response_str = line.get("response", "")
        first_response = extract_first_response(response_str)
        
        if first_response:
            predicted_=first_response.strip().rstrip(',').strip()  # Remove trailing commas/spaces
            predicte_t=predicted_.split("|")
            if len(predicte_t)==3:
                # Normalize entities (remove trailing punctuation, extra spaces)
                ph=normalize_entity(predicte_t[0].lower().replace("       "," ").strip())
                pt=normalize_entity(predicte_t[2].lower().replace("       "," ").strip())
                pr=predicte_t[1].lower().strip()
                P.add((ph,pr,pt))
  
        state_dict["p"] += len(P)
        state_dict["g"] += len(gold_triples)
        state_dict["c"] += len(P & gold_triples)

all_metirc_results = calclulate_f1(state_dict, 'all')

print("="*80)
print("CHEMPROT FIXED S-HRT MODEL (checkpoint 10000) EVALUATION RESULTS")
print("="*80)
print(all_metirc_results)
print(f"\nPrecision:  {all_metirc_results['all-prec']:.4f} ({all_metirc_results['all-prec']*100:.2f}%)")
print(f"Recall:     {all_metirc_results['all-recall']:.4f} ({all_metirc_results['all-recall']*100:.2f}%)")
print(f"F1-Score:   {all_metirc_results['all-f1']:.4f} ({all_metirc_results['all-f1']*100:.2f}%)")
print("="*80)
print(f"\nState dict: {state_dict}")
print(f"  Predictions: {state_dict['p']}")
print(f"  Ground truth: {state_dict['g']}")
print(f"  Correct: {state_dict['c']}")
print("="*80)

# Save results
with open("F1_EVALUATION_CHEMPROT_FIXED_S_HRT_10000.json", "w") as f:
    json.dump(all_metirc_results, f, indent=2)

with open("F1_EVALUATION_CHEMPROT_FIXED_S_HRT_10000.txt", "w") as f:
    f.write("CHEMPROT FIXED S-HRT MODEL (checkpoint 10000) EVALUATION RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Precision:  {all_metirc_results['all-prec']:.4f} ({all_metirc_results['all-prec']*100:.2f}%)\n")
    f.write(f"Recall:     {all_metirc_results['all-recall']:.4f} ({all_metirc_results['all-recall']*100:.2f}%)\n")
    f.write(f"F1-Score:   {all_metirc_results['all-f1']:.4f} ({all_metirc_results['all-f1']*100:.2f}%)\n")
    f.write("="*80 + "\n")
    f.write(f"State dict: {state_dict}\n")

print("\n✅ Results saved to F1_EVALUATION_CHEMPROT_FIXED_S_HRT_10000.json and F1_EVALUATION_CHEMPROT_FIXED_S_HRT_10000.txt")

