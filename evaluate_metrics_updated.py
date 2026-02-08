import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

# === Configuration ===
REF_PATH  = "config/reference.jsonl"
PRED_PATH = "config/prediction.jsonl"
THRESHOLD = 0.8  # cosine similarity threshold

# === Utility functions ===
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def embed_texts(texts, model):
    if not texts:
        return np.zeros((1, model.get_sentence_embedding_dimension()))
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# === Semantic evaluation ===
def semantic_eval(ref_items, pred_items, model, threshold=THRESHOLD):
    if not ref_items and not pred_items:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1, "recall": 1, "f1": 1}
    if not ref_items:
        return {"tp": 0, "fp": len(pred_items), "fn": 0, "precision": 0, "recall": 0, "f1": 0}
    if not pred_items:
        return {"tp": 0, "fp": 0, "fn": len(ref_items), "precision": 0, "recall": 0, "f1": 0}

    ref_emb = embed_texts(ref_items, model)
    pred_emb = embed_texts(pred_items, model)
    sim = cosine_similarity(ref_emb, pred_emb)
    tp, fn, matched_pred = 0, 0, set()

    for i in range(len(ref_items)):
        j = np.argmax(sim[i])
        if sim[i][j] >= threshold:
            tp += 1
            matched_pred.add(j)
        else:
            fn += 1

    fp = len(pred_items) - len(matched_pred)
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p,3), "recall": round(r,3), "f1": round(f1,3)}

# === Categorical evaluation ===
def categorical_eval(ref_labels, pred_labels):
    tp = sum([lab in pred_labels for lab in ref_labels])
    fp = sum([lab not in ref_labels for lab in pred_labels])
    fn = sum([lab not in pred_labels for lab in ref_labels])
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p,3), "recall": round(r,3), "f1": round(f1,3)}

# === Display helpers ===
def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

# === Main evaluation ===
def main():
    print("\n🔍 Loading reference and prediction files...")
    ref = load_jsonl(REF_PATH)
    pred = load_jsonl(PRED_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Loaded {len(ref)} reference records and {len(pred)} predictions.")

    summary = {
        "FP":[], "TEn":[], "FU":[], "DG":[], "SP":[],
        "Action":[], "Source":[], "Destination":[], "Movements":[]
    }

    for idx, (r, p) in enumerate(zip(ref, pred), start=1):
        fp = r.get("FP", f"FP_{idx}")

        # --- Textual ---
        fp_eval = semantic_eval([r.get("FP","")], [p.get("FP","")], model)
        ten_eval = semantic_eval([r.get("TEn","")], [p.get("TEn","")], model)
        fu_eval = semantic_eval([r.get("FU","")], [p.get("FU","")], model)
        dg_eval = semantic_eval(r.get("DG",[]), p.get("DG",[]), model)
        sp_eval = semantic_eval(r.get("SP",[]), p.get("SP",[]), model)

        # --- Categorical (Action / Source / Destination) ---
        act_eval = categorical_eval(r.get("Action",[]), p.get("Action",[]))
        src_eval = categorical_eval(r.get("Source",[]), p.get("Source",[]))
        dst_eval = categorical_eval(r.get("Destination",[]), p.get("Destination",[]))

        # --- Movements regroupés ---
        ref_mv = r.get("Movements", {})
        pred_mv = p.get("Movements", {})
        labels_ref = []
        labels_pred = []
        for mv, n in ref_mv.items():
            labels_ref += [mv] * int(n)
        for mv, n in pred_mv.items():
            labels_pred += [mv] * int(n)
        mv_eval = categorical_eval(labels_ref, labels_pred)

        # --- Stockage pour moyennes ---
        for k,v in {
            "FP":fp_eval,"TEn":ten_eval,"FU":fu_eval,"DG":dg_eval,"SP":sp_eval,
            "Action":act_eval,"Source":src_eval,"Destination":dst_eval,"Movements":mv_eval
        }.items():
            summary[k].append(v)

    # === Résumé global (style Table 4) ===
    print_section("📊 COSMIC key component identification (LLM + Rule-based)")
    table=[]
    for k in ["FP","TEn","FU","DG","SP","Action","Source","Destination","Movements"]:
        p=np.mean([x["precision"] for x in summary[k]])
        r=np.mean([x["recall"] for x in summary[k]])
        f=np.mean([x["f1"] for x in summary[k]])
        table.append([k, round(p,2), round(r,2), round(f,2)])
    table.append(["Average","-","-", round(np.mean([x[3] for x in table]),2)])
    print(tabulate(table, headers=["Component","Precision","Recall","F1"], tablefmt="grid"))

if __name__ == "__main__":
    main()
