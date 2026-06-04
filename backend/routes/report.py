"""
Report Router

Module 3 reporting endpoints: full model card, summary, narrative,
JSON export, and the cryptographic audit trail.
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from backend.services.pipeline_runner import get_cached
from backend.services.audit_log import get_log, get_chain_integrity

router = APIRouter(prefix="/report", tags=["Report"])


@router.get("")
@router.get("/")
def get_report():
    """Return the full IEEE model card as JSON."""
    card = get_cached("model_card")
    if card is None:
        return {"status": "no_results", "message": "Run the pipeline first."}
    return {"status": "ok", "data": card}


@router.get("/summary")
def get_report_summary():
    """Condensed report summary for the dashboard: risk, leakage, recommendations."""
    card = get_cached("model_card")
    if not card:
        return {"status": "no_results"}

    privacy = card.get("privacy_analysis", {})
    fairness = card.get("fairness", {})
    utility  = card.get("utility", {})

    return {
        "status":           "ok",
        "overall_risk":     privacy.get("overall_risk", "UNKNOWN"),
        "mean_leakage":     privacy.get("mean_leakage_score", 0),
        "recommendations":  card.get("recommendations", []),
        "narratives":       card.get("attribute_narratives", {}),
        "generated_at":     card.get("generated_at"),
        "audit_hash":       card.get("audit_hash"),
        "visualizations":   card.get("visualizations", {}),
        "mdd":              fairness.get("max_demographic_disparity", 0),
        "techniques":       card.get("mitigation", {}).get("techniques_applied", []),
        "tar_far":          utility,
    }


@router.get("/narratives")
def get_narratives():
    """Per-attribute natural-language narrative summaries."""
    card = get_cached("model_card")
    if not card:
        return {"status": "no_results"}
    return {"status": "ok", "data": card.get("attribute_narratives", {})}


@router.get("/download")
def download_model_card():
    """Download the full model card as a formatted JSON file."""
    card = get_cached("model_card")
    if not card:
        raise HTTPException(404, "No model card available. Run the pipeline first.")

    content = json.dumps(card, indent=2, default=str)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="obscura_model_card.json"',
            "Content-Type": "application/json; charset=utf-8",
        },
    )


@router.get("/audit-trail")
def get_audit_trail():
    """
    Return the full cryptographic audit trail with chain-integrity status.

    Each entry carries a SHA-256 hash over its content + prev entry hash,
    forming a tamper-evident chain.
    """
    integrity = get_chain_integrity()
    entries   = get_log()
    return {
        "status":      "ok",
        "integrity":   integrity,
        "entries":     entries,
        "total":       len(entries),
    }
