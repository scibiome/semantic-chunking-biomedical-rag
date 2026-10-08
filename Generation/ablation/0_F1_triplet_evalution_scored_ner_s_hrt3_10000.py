import json
import re

def normalize_entity(text):
    text = re.sub(r'[.,;:!?)\]}\]]+$', '', text)
    text = ' '.join(text.split())
    return text.strip()

state_dict = {"p": 0, "c": 0, "g": 0}
def calclulate_f1(statics_dict, prefix=""):
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}


i=-1
with open("/home/users/riah/project_thesis/BIoMedRAG/4_generation_triple_model/test_triplet_chunk_scorer_scored_ner_s_hrt3_10000.json", "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        line=line.strip()
        line=json.loads(line)
        i=i+1
        P=set()
        gold_triples=set()
        ground_truth=line["ground_truth"]
        gold_t=ground_truth.split("|")
        if len(gold_t) != 3:
            continue
        gh=normalize_entity(gold_t[0].lower().replace("       "," ").strip())
        gr=gold_t[1].lower().strip()
        gt=normalize_entity(gold_t[2].lower().replace("       "," ").strip())
        gold_triples.add((gh,gr,gt))
        predictions=line["response"].split("\n\n")[0].split("\n")
        if len(predictions)>=1:
            predicted_=predictions[0].strip().rstrip(',').strip()
            predicte_t=predicted_.split("|")
            if len(predicte_t)==3:
                ph=normalize_entity(predicte_t[0].lower().replace("       "," ").strip())
                pt=normalize_entity(predicte_t[2].lower().replace("       "," ").strip())
                pr=predicte_t[1].lower().strip()
                P.add((ph,pr,pt))
        state_dict["p"] += len(P)
        state_dict["g"] += len(gold_triples)
        state_dict["c"] += len(P & gold_triples)

all_metirc_results = calclulate_f1(state_dict, 'all')

print("="*80)
print("SCORED NER S-HRT3 MODEL (checkpoint 10000) EVALUATION RESULTS")
print("="*80)
print(all_metirc_results)
print(f"\nPrecision:  {all_metirc_results['all-prec']:.4f}")
print(f"Recall:     {all_metirc_results['all-recall']:.4f}")
print(f"F1-Score:   {all_metirc_results['all-f1']:.4f}")
print("="*80)

with open("F1_EVALUATION_SCORED_NER_S_HRT3_10000.json", "w") as f:
    json.dump(all_metirc_results, f, indent=2)
with open("F1_EVALUATION_SCORED_NER_S_HRT3_10000.txt", "w") as f:
    f.write("SCORED NER S-HRT3 MODEL (checkpoint 10000) EVALUATION RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Precision:  {all_metirc_results['all-prec']:.4f}\n")
    f.write(f"Recall:     {all_metirc_results['all-recall']:.4f}\n")
    f.write(f"F1-Score:   {all_metirc_results['all-f1']:.4f}\n")
    f.write("="*80 + "\n")
    f.write(f"State dict: {state_dict}\n")
print("\nResults saved to F1_EVALUATION_SCORED_NER_S_HRT3_10000.json and F1_EVALUATION_SCORED_NER_S_HRT3_10000.txt")
