"""Report router — model card and fairness report."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from backend.services.pipeline_runner import get_cached

router = APIRouter(prefix="/report", tags=["Report"])


@router.get("")
@router.get("/")
def get_report():
    """Return full model card report."""
    card = get_cached("model_card")
    if card is None:
        return {"status": "no_results", "message": "Run the pipeline first."}
    return {"status": "ok", "data": card}


@router.get("/summary")
def get_report_summary():
    """Return condensed report summary for the dashboard."""
    card = get_cached("model_card")
    if not card:
        return {"status": "no_results"}

    return {
        "status": "ok",
        "overall_risk":       card.get("privacy_analysis", {}).get("overall_risk"),
        "mean_leakage":       card.get("privacy_analysis", {}).get("mean_leakage_score"),
        "recommendations":    card.get("recommendations", []),
        "narratives":         card.get("attribute_narratives", {}),
        "generated_at":       card.get("generated_at"),
        "visualizations":     card.get("visualizations", {}),
        "mdd":                card.get("fairness", {}).get("max_demographic_disparity", 0),
        "techniques":         card.get("mitigation", {}).get("techniques_applied", []),
    }


@router.get("/narratives")
def get_narratives():
    card = get_cached("model_card")
    if not card:
        return {"status": "no_results"}
    return {"status": "ok", "data": card.get("attribute_narratives", {})}
