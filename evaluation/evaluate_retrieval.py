import os
import sys
import json
import numpy as np

# Ensure UTF-8 console encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retrieval.hybrid import HybridRetriever


def evaluate_retrieval(
    dataset_path: str = "evaluation/eval_dataset.json",
    top_k_list: list = [1, 3, 5]
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        eval_cases = json.load(f)

    print("==================================================")
    print("VIDHISETU HYBRID RETRIEVAL EVALUATION")
    print("==================================================")

    retriever = HybridRetriever()
    retriever.initialize()

    recalls = {k: [] for k in top_k_list}
    mrr_list = []
    category_metrics = {}

    for case in eval_cases:
        q_id = case["id"]
        category = case["category"]
        query = case["query"]
        exp_act = case.get("expected_act")
        exp_sec = case.get("expected_section")

        if category not in category_metrics:
            category_metrics[category] = {"total": 0, "mrr": [], "hits_at_1": 0}

        category_metrics[category]["total"] += 1

        results = retriever.search(query, top_k=max(top_k_list))

        # Check absent evidence case
        if exp_act is None and exp_sec is None:
            # For absent evidence, success means no false positive matches with high exact match flag
            has_false_positive = any(r.get("exact_match", False) for r in results)
            mrr_val = 1.0 if not has_false_positive else 0.0
            for k in top_k_list:
                recalls[k].append(1.0 if not has_false_positive else 0.0)
            mrr_list.append(mrr_val)
            category_metrics[category]["mrr"].append(mrr_val)
            if not has_false_positive:
                category_metrics[category]["hits_at_1"] += 1
            print(f"[{q_id:3s}] {category:16s} | Query: \"{query[:45]:45s}...\" | ABSENT EVIDENCE PASS")
            continue

        # Evaluate rank of first matching result
        first_hit_rank = None
        for rank, res in enumerate(results, 1):
            r_act = res.get("act", "").lower()
            r_sec = res.get("section_number")

            act_match = True
            if exp_act:
                act_match = exp_act.lower() in r_act or exp_act.lower() in res.get("source", "").lower() or exp_act.lower() in res.get("chunk_id", "").lower()

            if isinstance(exp_sec, list):
                sec_match = (r_sec in exp_sec)
            else:
                sec_match = (exp_sec is None) or (r_sec == exp_sec)

            if act_match and sec_match:
                first_hit_rank = rank
                break

        # Record MRR
        if first_hit_rank is not None:
            mrr_val = 1.0 / first_hit_rank
            if first_hit_rank == 1:
                category_metrics[category]["hits_at_1"] += 1
        else:
            mrr_val = 0.0

        mrr_list.append(mrr_val)
        category_metrics[category]["mrr"].append(mrr_val)

        # Record Recall@K
        for k in top_k_list:
            hit = 1.0 if (first_hit_rank is not None and first_hit_rank <= k) else 0.0
            recalls[k].append(hit)

        rank_str = f"Rank {first_hit_rank}" if first_hit_rank else "NOT FOUND"
        print(f"[{q_id:3s}] {category:16s} | Query: \"{query[:45]:45s}\" -> {rank_str}")

    mean_mrr = float(np.mean(mrr_list))
    mean_recalls = {k: float(np.mean(recalls[k])) for k in top_k_list}

    print("\n==================================================")
    print("SUMMARY EVALUATION METRICS")
    print("==================================================")
    for k in top_k_list:
        print(f"Recall@{k}:  {mean_recalls[k]:.4f} ({mean_recalls[k]*100:.1f}%)")
    print(f"MRR:       {mean_mrr:.4f}")

    print("\n--- CATEGORY BREAKDOWN ---")
    for cat, stats in category_metrics.items():
        cat_mrr = float(np.mean(stats["mrr"])) if stats["mrr"] else 0.0
        acc = (stats["hits_at_1"] / stats["total"]) * 100
        print(f"{cat:18s} | Count: {stats['total']} | Top-1 Accuracy: {acc:5.1f}% | MRR: {cat_mrr:.4f}")

    print("==================================================\n")

    return {
        "mrr": mean_mrr,
        "recalls": mean_recalls,
        "category_metrics": category_metrics
    }


if __name__ == "__main__":
    evaluate_retrieval()
