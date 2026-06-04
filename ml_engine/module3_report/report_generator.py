"""
Module 3: Report Generator

Produces the IEEE model card–style privacy report in JSON format,
including SHA-256 cryptographic signatures over the audit artifacts.
"""

import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import RESULTS_DIR, ATTRIBUTES
from ml_engine.module3_report.metrics_engine import (
    compute_privacy_metrics, build_model_card
)


def _sha256_report(data: dict) -> str:
    """Compute a deterministic SHA-256 over the structured audit data."""
    def _convert(obj):
        if hasattr(obj, "item"):   return obj.item()   # numpy scalar
        if hasattr(obj, "tolist"): return obj.tolist()  # numpy array
        return obj

    def _recurse(obj):
        if isinstance(obj, dict):  return {k: _recurse(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):  return [_recurse(i) for i in obj]
        return _convert(obj)

    normalised = _recurse(data)
    raw = json.dumps(normalised, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def generate_full_report(
    audit_baseline: dict,
    mitigation_results: dict,
    fairness_results: dict,
    plot_paths: dict,
    dataset_info: dict = None,
    save: bool = True,
) -> dict:
    """
    Assemble the complete privacy report with cryptographic audit hash.

    Returns
    -------
    Full report dict (IEEE model card format) with `audit_hash` field.
    """
    if dataset_info is None:
        dataset_info = {
            "name":       "Synthetic CelebA-like Dataset",
            "n_samples":  12000,
            "embedding_dim": 512,
            "attributes": list(ATTRIBUTES.keys()),
            "source":     "Synthetically generated with realistic attribute correlations",
            "reference":  "Wang et al. (2019) Race Faces in the Wild, ICCVW",
        }

    audit_post_adv = mitigation_results.get("adv_results", {})

    model_card = build_model_card(
        dataset_info=dataset_info,
        audit_results=audit_baseline,
        mitigation_results=mitigation_results,
        fairness_results=fairness_results,
    )

    # ── Cryptographic audit hash ──────────────────────────────────────────────
    # Hash covers the core audit + mitigation data only (not timestamps/plots)
    audit_payload = {
        "baseline_summary": audit_baseline.get("__summary__", {}),
        "comparison":       mitigation_results.get("comparison", []),
        "fairness_mdd":     fairness_results.get("max_demographic_disparity", 0),
        "dataset":          dataset_info,
    }
    audit_hash = _sha256_report(audit_payload)

    model_card["generated_at"]    = datetime.utcnow().isoformat() + "Z"
    model_card["audit_hash"]       = audit_hash
    model_card["visualizations"]   = {
        k: str(Path(v).name) for k, v in plot_paths.items() if v
    }
    model_card["attribute_narratives"] = _generate_narratives(audit_baseline, audit_post_adv)

    if save:
        out = RESULTS_DIR / "model_card.json"
        _save_json(model_card, out)
        print(f"[Report] Model card saved to {out}")
        print(f"[Report] Audit SHA-256: {audit_hash}")

    return model_card


def _generate_narratives(baseline: dict, post: dict) -> dict:
    narratives = {}
    for attr in ATTRIBUTES:
        if attr not in baseline:
            continue
        b   = baseline[attr]
        p   = post.get(attr, {})

        b_auc     = b["best_attacker_auc"]
        p_auc     = p.get("best_attacker_auc", b_auc)
        reduction = (b_auc - p_auc) / (b_auc + 1e-8) * 100
        risk      = b["risk_level"]
        label     = attr.replace("_", " ").title()

        narratives[attr] = (
            f"{label} leakage classified as {risk} with AUC={b_auc:.3f} "
            f"(LR={b['logistic_regression']['auc_roc']:.3f}, "
            f"MLP={b['mlp']['auc_roc']:.3f}). "
            f"After adversarial disentanglement, AUC reduced to {p_auc:.3f} "
            f"(↓{reduction:.1f}% reduction). "
            f"Leakage score: {b['best_leakage_score']:.3f} → "
            f"{p.get('best_leakage_score', b['best_leakage_score']):.3f}."
        )
    return narratives


def _save_json(data: dict, path: Path):
    import numpy as np

    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj

    def _recurse(obj):
        if isinstance(obj, dict):  return {k: _recurse(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_recurse(i) for i in obj]
        return _convert(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_recurse(data), f, indent=2)
