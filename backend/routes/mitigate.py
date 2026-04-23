"""Mitigation router — Module 2 endpoints."""
from fastapi import APIRouter
from backend.services.pipeline_runner import get_cached

router = APIRouter(prefix="/mitigate", tags=["Mitigation"])


@router.get("")
@router.get("/")
def get_mitigation_results():
    data = get_cached("mitigation")
    if data is None:
        return {"status": "no_results", "message": "Run the pipeline first."}
    return {"status": "ok", "data": data}


@router.get("/comparison")
def get_comparison():
    """Before/after AUC comparison table."""
    data = get_cached("mitigation")
    if not data:
        return {"status": "no_results"}
    return {
        "status":     "ok",
        "comparison": data.get("comparison", []),
    }


@router.get("/sweep/adversarial")
def get_adv_sweep():
    data = get_cached("mitigation")
    if not data:
        return {"status": "no_results"}
    return {"status": "ok", "sweep": data.get("adv_sweep", [])}


@router.get("/sweep/noise")
def get_noise_sweep():
    data = get_cached("mitigation")
    if not data:
        return {"status": "no_results"}
    return {"status": "ok", "sweep": data.get("noise_sweep", [])}
