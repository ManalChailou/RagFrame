import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import re
import ast

# === Configuration ===
REF_PATH  = "config/True_reference_business_Restosys.jsonl"
# REF_PATH = "config/True_reference_business copy.jsonl"

PATTERN_CONFIG_PATH = "config/cosmic_patterns.json"
PRED_PATH = "config/prediction.jsonl"
THRESHOLD = 0.8


# === Utility functions ===
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_texts(texts, model):
    texts = [str(t) for t in texts if str(t).strip()]
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def flatten_dg(dg):
    """
    Convert DG from prediction/reference format:
    {
      "ObjectOfInterest": {
        "DataGroupName": ["attr1", "attr2"]
      }
    }

    into semantic strings:
    [
      "ObjectOfInterest | DataGroupName | attr1, attr2"
    ]
    """
    items = []

    if not dg:
        return items

    # Already list format
    if isinstance(dg, list):
        for x in dg:
            if isinstance(x, str):
                items.append(x)
            elif isinstance(x, dict):
                # Supports old or mixed formats
                ooi = x.get("ObjectOfInterest") or x.get("object_of_interest") or x.get("name") or ""
                data_group = (
                    x.get("DataGroup")
                    or x.get("DataGroupName")
                    or x.get("data_group")
                    or x.get("data_group_name")
                    or ""
                )
                attrs = x.get("Attributes") or x.get("attributes") or []

                if data_group:
                    items.append(f"{ooi} | {data_group} | {', '.join(map(str, attrs))}")
                else:
                    for key, value in x.items():
                        if key in ["name", "ObjectOfInterest", "Attributes", "attributes"]:
                            continue
                        if isinstance(value, list):
                            items.append(f"{ooi} | {key} | {', '.join(map(str, value))}")
                        elif isinstance(value, dict):
                            attrs = value.get("Attributes") or value.get("attributes") or []
                            items.append(f"{ooi} | {key} | {', '.join(map(str, attrs))}")
        return items

    # New dict format
    if isinstance(dg, dict):
        for ooi, groups in dg.items():
            if not isinstance(groups, dict):
                continue

            for dg_name, attrs in groups.items():
                if dg_name in ["ObjectOfInterest", "Attributes", "attributes"]:
                    continue

                if isinstance(attrs, dict):
                    attrs = attrs.get("Attributes") or attrs.get("attributes") or []
                elif not isinstance(attrs, list):
                    attrs = [attrs]

                items.append(f"{ooi} | {dg_name} | {', '.join(map(str, attrs))}")

    return items



# === Semantic evaluation ===
def semantic_eval(ref_items, pred_items, model, threshold=THRESHOLD):
    ref_items = [str(x).strip() for x in ref_items if str(x).strip()]
    pred_items = [str(x).strip() for x in pred_items if str(x).strip()]

    if not ref_items and not pred_items:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1, "recall": 1, "f1": 1}
    if not ref_items:
        return {"tp": 0, "fp": len(pred_items), "fn": 0, "precision": 0, "recall": 0, "f1": 0}
    if not pred_items:
        return {"tp": 0, "fp": 0, "fn": len(ref_items), "precision": 0, "recall": 0, "f1": 0}

    ref_emb = embed_texts(ref_items, model)
    pred_emb = embed_texts(pred_items, model)

    sim = cosine_similarity(ref_emb, pred_emb)

    # Greedy one-to-one matching
    pairs = []
    for i in range(len(ref_items)):
        for j in range(len(pred_items)):
            pairs.append((sim[i][j], i, j))

    pairs.sort(reverse=True)

    matched_ref = set()
    matched_pred = set()

    for score, i, j in pairs:
        if score >= threshold and i not in matched_ref and j not in matched_pred:
            matched_ref.add(i)
            matched_pred.add(j)

    tp = len(matched_ref)
    fp = len(pred_items) - len(matched_pred)
    fn = len(ref_items) - len(matched_ref)

    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3)
    }


# === Categorical evaluation ===
def categorical_eval(ref_labels, pred_labels):
    """
    Multiset evaluation.
    Correctly handles repeated labels such as:
    ["Entry", "Entry", "Exit"]
    """
    ref_counter = Counter(ref_labels)
    pred_counter = Counter(pred_labels)

    tp = sum(min(ref_counter[k], pred_counter[k]) for k in ref_counter)
    fp = sum(max(pred_counter[k] - ref_counter[k], 0) for k in pred_counter)
    fn = sum(max(ref_counter[k] - pred_counter[k], 0) for k in ref_counter)

    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3)
    }


def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

import ast
import re

def normalize_text(text):
    """
    Generic text normalization:
    - lowercase
    - replace underscores/hyphens with spaces
    - remove extra spaces
    - remove simple punctuation
    No domain-specific hard coding.
    """
    text = str(text or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_list(value):
    """
    Generic conversion to list.
    Handles:
    - "Timer"
    - ["Timer", "Heater"]
    - stringified list: "['Timer', 'Heater']"
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()

        # If the list was accidentally saved as a string
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        return [value]

    return [value]

def normalize_component_list(value):
    """
    Generic list normalization for components such as FU.
    No use-case-specific mapping.
    """
    items = ensure_list(value)

    normalized = []
    for item in items:
        item = normalize_text(item)
        if item and item not in normalized:
            normalized.append(item)

    return normalized


def load_pattern_config(path=PATTERN_CONFIG_PATH):
    """
    Load cosmic_patterns.json.
    If unavailable, return an empty config so the evaluator still runs.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"entity_patterns": {}, "movement_rules": {}}


def _regex_matches(pattern, text):
    try:
        return re.search(pattern, str(text or ""), flags=0) is not None
    except re.error:
        return False


def _entity_candidates(entity):
    """
    Create generic candidates for pattern matching.
    This allows patterns like ^ram$ to match labels such as 'Current settings (RAM)'.
    """
    original = str(entity or "").strip()
    norm = normalize_text(original)
    parts = norm.split()

    return [original, norm] + parts


def normalize_entity_category(entity, config):
    """
    Generic entity category normalization using cosmic_patterns.json.
    """
    patterns = config.get("entity_patterns", {})

    ordered = [
        ("external", "External Application"),
        ("temporal", "Clock"),
        ("storage", "Storage"),
        ("user", "User"),
        ("system", "System"),
    ]

    candidates = _entity_candidates(entity)

    for raw_category, normalized_category in ordered:
        for pattern in patterns.get(raw_category, []):
            if any(_regex_matches(pattern, c) for c in candidates):
                return normalized_category

    return str(entity or "").strip().title()


def normalize_action_with_patterns(action, source="", destination="", config=None):
    """
    Normalize Action using cosmic_patterns.json movement_rules.
    Example:
    - Enter -> Entry
    - Store -> Write
    - Get/Retrieve -> Read
    - Send/Display -> Exit

    This is generic because it uses the configured movement rules and COSMIC directions.
    """
    config = config or {"movement_rules": {}, "entity_patterns": {}}

    action_raw = str(action or "").strip()
    action_norm = normalize_text(action_raw)

    if not action_norm:
        return ""

    canonical = {
        "entry": "Entry",
        "read": "Read",
        "write": "Write",
        "exit": "Exit",
        "internal": "Internal",
    }

    if action_norm in canonical:
        return canonical[action_norm]

    src_cat = normalize_entity_category(source, config)
    dst_cat = normalize_entity_category(destination, config)

    # 1. Use configured movement rules with source/destination context.
    for movement_type, rules in config.get("movement_rules", {}).items():
        for rule in rules:
            if len(rule) < 3:
                continue

            pattern, expected_src, expected_dst = rule[0], rule[1], rule[2]

            if (
                src_cat == expected_src
                and dst_cat == expected_dst
                and _regex_matches(pattern, action_raw)
            ):
                return movement_type

    # 2. Generic COSMIC direction fallback.
    if src_cat in {"User", "External Application", "Clock"} and dst_cat == "System":
        return "Entry"

    if src_cat == "Storage" and dst_cat == "System":
        return "Read"

    if src_cat == "System" and dst_cat == "Storage":
        return "Write"

    if src_cat == "System" and dst_cat in {"User", "External Application"}:
        return "Exit"

    # 3. Action-only configured fallback for unambiguous verbs.
    matches = []

    for movement_type, rules in config.get("movement_rules", {}).items():
        for rule in rules:
            if len(rule) >= 1 and _regex_matches(rule[0], action_raw):
                matches.append(movement_type)

    unique_matches = list(dict.fromkeys(matches))

    if len(unique_matches) == 1:
        return unique_matches[0]

    return action_raw.title()


def normalize_action_sequence(actions, sources=None, destinations=None, config=None):
    actions = ensure_list(actions)
    sources = ensure_list(sources)
    destinations = ensure_list(destinations)

    normalized = []

    for i, action in enumerate(actions):
        src = sources[i] if i < len(sources) else ""
        dst = destinations[i] if i < len(destinations) else ""

        label = normalize_action_with_patterns(action, src, dst, config)

        if label:
            normalized.append(label)

    return normalized

def normalize_item_list(value):
    """
    Convert any component field into a clean list of normalized strings.
    Generic: works for FU, Source, Destination, etc.
    """
    items = ensure_list(value)

    normalized = []
    for item in items:
        item = normalize_text(item)
        if item and item not in normalized:
            normalized.append(item)

    return normalized

# === Main evaluation ===
def main():
    print("\n🔍 Loading reference and prediction files...")

    ref = load_jsonl(REF_PATH)
    pred = load_jsonl(PRED_PATH)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    pattern_config = load_pattern_config(PATTERN_CONFIG_PATH)

    print(f"Loaded {len(ref)} reference records and {len(pred)} predictions.")

    summary = {
        "FP": [],
        "TEn": [],
        "FU": [],
        "DG": [],
        "SP": [],
        "Action": [],
        "Source": [],
        "Destination": [],
        "Movements": []
    }

    for idx, (r, p) in enumerate(zip(ref, pred), start=1):
        # --- Textual / semantic ---
        fp_eval = semantic_eval([r.get("FP", "")], [p.get("FP", "")], model)
        ten_eval = semantic_eval([r.get("TEn", "")], [p.get("TEn", "")], model)
        fu_eval = semantic_eval(
            normalize_item_list(r.get("FU", [])),
            normalize_item_list(p.get("FU", [])),
            model,
            threshold=0.75
        )
        dg_eval = semantic_eval(
            flatten_dg(r.get("DG", {})),
            flatten_dg(p.get("DG", {})),
            model,
            threshold=0.75
        )

        sp_eval = semantic_eval(
            r.get("SP", []),
            p.get("SP", []),
            model,
            threshold=0.65
        )

        # --- Categorical ---
        act_eval = categorical_eval(
            normalize_action_sequence(
                r.get("Action", []),
                r.get("Source", []),
                r.get("Destination", []),
                pattern_config
            ),
            normalize_action_sequence(
                p.get("Action", []),
                p.get("Source", []),
                p.get("Destination", []),
                pattern_config
            )
        )
        src_eval = categorical_eval(r.get("Source", []), p.get("Source", []))
        dst_eval = categorical_eval(r.get("Destination", []), p.get("Destination", []))

        # --- Movement counts ---
        ref_mv = r.get("Movements", {})
        pred_mv = p.get("Movements", {})

        labels_ref = []
        labels_pred = []

        for mv, n in ref_mv.items():
            labels_ref += [mv] * int(n)

        for mv, n in pred_mv.items():
            labels_pred += [mv] * int(n)

        mv_eval = categorical_eval(labels_ref, labels_pred)

        for k, v in {
            "FP": fp_eval,
            "TEn": ten_eval,
            "FU": fu_eval,
            "DG": dg_eval,
            "SP": sp_eval,
            "Action": act_eval,
            "Source": src_eval,
            "Destination": dst_eval,
            "Movements": mv_eval
        }.items():
            summary[k].append(v)

    # === Global summary ===
    print_section("📊 COSMIC key component identification")

    table = []

    for k in [
        "FP",
        "TEn",
        "FU",
        "DG",
        "SP",
        "Action",
        "Source",
        "Destination",
        "Movements"
    ]:
        precision = np.mean([x["precision"] for x in summary[k]])
        recall = np.mean([x["recall"] for x in summary[k]])
        f1 = np.mean([x["f1"] for x in summary[k]])

        table.append([
            k,
            round(precision, 2),
            round(recall, 2),
            round(f1, 2)
        ])

    avg_f1 = round(np.mean([row[3] for row in table]), 2)
    table.append(["Average", "-", "-", avg_f1])

    print(tabulate(
        table,
        headers=["Component", "Precision", "Recall", "F1"],
        tablefmt="grid"
    ))


if __name__ == "__main__":
    main()