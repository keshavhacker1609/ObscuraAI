"""Metrics router — aggregated privacy, utility, fairness metrics."""
from fastapi import APIRouter
from backend.services.pipeline_runner import get_cached

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/audit")
def audit_metrics():
    """Structured privacy metrics from audit."""
    audit = get_cached("audit")
    if not audit:
        return {"status": "no_results"}

    attrs = {}
    for attr in ["gender", "age_group", "ethnicity"]:
        if attr in audit:
            d = audit[attr]
            attrs[attr] = {
                "lr":    d.get("logistic_regression", {}),
                "mlp":   d.get("mlp", {}),
                "risk":  d.get("risk_level"),
                "best_auc":       d.get("best_attacker_auc"),
                "leakage_score":  d.get("best_leakage_score"),
            }

    return {
        "status":     "ok",
        "attributes": attrs,
        "summary":    audit.get("__summary__", {}),
    }


@router.get("/fairness")
def fairness_metrics():
    """Fairness metrics — per-demographic leakage disparity."""
    fairness = get_cached("fairness")
    if not fairness:
        return {"status": "no_results"}
    return {"status": "ok", "data": fairness}


@router.get("/utility")
def utility_metrics():
    """Face verification utility (TAR@FAR)."""
    tar_far = get_cached("tar_far")
    if not tar_far:
        return {"status": "no_results"}
    return {"status": "ok", "data": tar_far}


@router.get("/overview")
def overview():
    """Single endpoint for dashboard overview — all key numbers."""
    audit   = get_cached("audit")    or {}
    mitig   = get_cached("mitigation") or {}
    fairness = get_cached("fairness") or {}
    tar_far  = get_cached("tar_far")  or {}

    summary = audit.get("__summary__", {})

    # Compute overall improvement
    comparison = mitig.get("comparison", [])
    avg_adv_reduction = 0.0
    if comparison:
        avg_adv_reduction = sum(r["adv_reduction_pct"] for r in comparison) / len(comparison)

    return {
        "status": "ok",
        "overall_risk":           summary.get("overall_risk_level", "UNKNOWN"),
        "mean_leakage_baseline":  summary.get("mean_leakage_score", 0),
        "max_auc_baseline":       summary.get("max_attacker_auc", 0),
        "avg_adv_reduction_pct":  round(avg_adv_reduction, 2),
        "mdd":                    fairness.get("max_demographic_disparity", 0),
        "mean_genuine_similarity": tar_far.get("mean_genuine_similarity", 0),
        "n_attributes":           summary.get("n_attributes_audited", 0),
        "pipeline_status":        "completed" if audit else "idle",
    }
