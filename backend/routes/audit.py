"""Audit Router — Module 1 endpoints."""

import numpy as np
from fastapi import APIRouter
from backend.services.pipeline_runner import get_cached

router = APIRouter(prefix="/audit", tags=["Audit"])

_ATTRIBUTES = ["gender", "age_group", "ethnicity"]


@router.get("")
@router.get("/")
def get_audit_results():
    """Return the full baseline audit results."""
    data = get_cached("audit")
    if data is None:
        return {
            "status":  "no_results",
            "message": "No audit results yet. POST /pipeline/run to start.",
        }
    return {"status": "ok", "data": data}


@router.get("/summary")
def get_audit_summary():
    """Return high-level audit summary (risk levels, leakage scores)."""
    data = get_cached("audit")
    if not data:
        return {"status": "no_results"}

    attrs = {}
    for attr in _ATTRIBUTES:
        if attr in data:
            d = data[attr]
            attrs[attr] = {
                "best_auc":          d.get("best_attacker_auc", 0),
                "leakage_score":     d.get("best_leakage_score", 0),
                "risk_level":        d.get("risk_level", "UNKNOWN"),
                "lr_auc":            d.get("logistic_regression", {}).get("auc_roc", 0),
                "mlp_auc":           d.get("mlp", {}).get("auc_roc", 0),
                "balanced_accuracy": d.get("mlp", {}).get("balanced_accuracy", 0),
                "mutual_information": d.get("mlp", {}).get("mutual_information", 0),
            }

    summary = data.get("__summary__", {})
    return {
        "status":       "ok",
        "attributes":   attrs,
        "overall_risk": summary.get("overall_risk_level", "UNKNOWN"),
        "mean_leakage": summary.get("mean_leakage_score", 0),
        "max_auc":      summary.get("max_attacker_auc", 0),
    }


@router.get("/roc-data")
def get_roc_data():
    """
    Return parametric ROC curve points for each attribute (baseline + post-adv).

    Uses the same parametric envelope method as the matplotlib visualizer:
      tpr = fpr^(1/k),  k = AUC / (1 − AUC)

    Returns 100-point curves suitable for rendering in Recharts.
    """
    audit    = get_cached("audit")
    mitig    = get_cached("mitigation")

    if not audit:
        return {"status": "no_results"}

    def _roc_points(auc: float, n: int = 100):
        k = max(0.01, auc / (1 - auc + 1e-8))
        fpr = np.linspace(0, 1, n)
        tpr = np.clip(np.power(np.maximum(fpr, 0), 1.0 / k), 0, 1)
        tpr[0] = 0.0; tpr[-1] = 1.0
        return [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
                for f, t in zip(fpr, tpr)]

    adv_results = (mitig or {}).get("adv_results", {})

    curves = {}
    for attr in _ATTRIBUTES:
        if attr not in audit:
            continue
        b_auc = audit[attr].get("best_attacker_auc", 0.5)
        p_auc = adv_results.get(attr, {}).get("best_attacker_auc", b_auc)
        curves[attr] = {
            "baseline":        _roc_points(b_auc),
            "post_mitigation": _roc_points(p_auc),
            "baseline_auc":    round(b_auc, 4),
            "post_auc":        round(p_auc, 4),
        }

    return {"status": "ok", "curves": curves}


@router.get("/per-class")
def get_per_class_metrics():
    """
    Return per-class precision/recall/F1 from the MLP classification report.
    Useful for granular fairness analysis beyond aggregate AUC.
    """
    audit = get_cached("audit")
    if not audit:
        return {"status": "no_results"}

    per_class = {}
    for attr in _ATTRIBUTES:
        if attr not in audit:
            continue
        report = audit[attr].get("classification_report", {})
        labels = audit[attr].get("labels", [])
        rows = []
        for i, label in enumerate(labels):
            key = str(i)
            if key in report:
                r = report[key]
                rows.append({
                    "class":     label,
                    "precision": round(r.get("precision", 0), 4),
                    "recall":    round(r.get("recall", 0), 4),
                    "f1_score":  round(r.get("f1-score", 0), 4),
                    "support":   r.get("support", 0),
                })
        per_class[attr] = rows

    return {"status": "ok", "per_class": per_class}
