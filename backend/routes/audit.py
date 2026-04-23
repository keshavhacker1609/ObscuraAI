"""Audit router — Module 1 endpoints."""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from backend.services.pipeline_runner import (
    get_cached, get_status, start_pipeline
)

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get("")
@router.get("/")
def get_audit_results():
    """Return the full baseline audit results."""
    data = get_cached("audit")
    if data is None:
        return {
            "status": "no_results",
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
    for attr in ["gender", "age_group", "ethnicity"]:
        if attr in data:
            d = data[attr]
            attrs[attr] = {
                "best_auc":          d.get("best_attacker_auc", 0),
                "leakage_score":     d.get("best_leakage_score", 0),
                "risk_level":        d.get("risk_level", "UNKNOWN"),
                "lr_auc":            d.get("logistic_regression", {}).get("auc_roc", 0),
                "mlp_auc":           d.get("mlp", {}).get("auc_roc", 0),
                "balanced_accuracy": d.get("mlp", {}).get("balanced_accuracy", 0),
            }

    summary = data.get("__summary__", {})
    return {
        "status":        "ok",
        "attributes":    attrs,
        "overall_risk":  summary.get("overall_risk_level", "UNKNOWN"),
        "mean_leakage":  summary.get("mean_leakage_score", 0),
        "max_auc":       summary.get("max_attacker_auc", 0),
    }
