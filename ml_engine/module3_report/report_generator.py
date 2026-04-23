"""
Module 3: Report Generator

Produces the IEEE model card–style privacy report in JSON format,
and optionally PDF via reportlab.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import RESULTS_DIR, ATTRIBUTES
from ml_engine.module3_report.metrics_engine import (
    compute_privacy_metrics, build_model_card
)


def generate_full_report(
    audit_baseline: dict,
    mitigation_results: dict,
    fairness_results: dict,
    plot_paths: dict,
    dataset_info: dict = None,
    save: bool = True,
) -> dict:
    """
    Assemble the complete privacy report.

    Returns
    -------
    Full report dict (model card format)
    """
    if dataset_info is None:
        dataset_info = {
            "name": "Synthetic CelebA-like Dataset",
            "n_samples": 12000,
            "embedding_dim": 512,
            "attributes": list(ATTRIBUTES.keys()),
            "source": "Synthetically generated with realistic attribute correlations",
            "reference": "Wang et al. (2019) Race Faces in the Wild, ICCVW",
        }

    audit_post_adv = mitigation_results.get("adv_results", {})

    model_card = build_model_card(
        dataset_info=dataset_info,
        audit_results=audit_baseline,
        mitigation_results=mitigation_results,
        fairness_results=fairness_results,
    )

    # Enrich with timestamps and visualizations
    model_card["generated_at"]  = datetime.utcnow().isoformat() + "Z"
    model_card["visualizations"] = {
        k: str(Path(v).name) for k, v in plot_paths.items() if v
    }

    # Per-attribute narrative summary
    model_card["attribute_narratives"] = _generate_narratives(audit_baseline, audit_post_adv)

    if save:
        out = RESULTS_DIR / "model_card.json"
        _save_json(model_card, out)
        print(f"[Report] Model card saved to {out}")

    return model_card


def _generate_narratives(baseline: dict, post: dict) -> dict:
    narratives = {}
    for attr in ATTRIBUTES:
        if attr not in baseline:
            continue
        b = baseline[attr]
        p = post.get(attr, {})

        b_auc = b["best_attacker_auc"]
        p_auc = p.get("best_attacker_auc", b_auc)
        reduction = (b_auc - p_auc) / (b_auc + 1e-8) * 100

        risk = b["risk_level"]
        label = attr.replace("_", " ").title()

        narratives[attr] = (
            f"{label} leakage was classified as {risk} with AUC={b_auc:.3f} "
            f"(LR={b['logistic_regression']['auc_roc']:.3f}, MLP={b['mlp']['auc_roc']:.3f}). "
            f"After adversarial disentanglement, AUC reduced to {p_auc:.3f} "
            f"(↓{reduction:.1f}% reduction). "
            f"Leakage score: {b['best_leakage_score']:.3f} → {p.get('best_leakage_score', b['best_leakage_score']):.3f}."
        )
    return narratives


def _save_json(data: dict, path: Path):
    import numpy as np

    def _convert(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return obj

    def _recurse(obj):
        if isinstance(obj, dict):  return {k: _recurse(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_recurse(i) for i in obj]
        return _convert(obj)

    with open(path, "w") as f:
        json.dump(_recurse(data), f, indent=2)
