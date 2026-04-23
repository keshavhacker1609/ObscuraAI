"""Pipeline router — start, status, reset."""
from fastapi import APIRouter
from backend.services.pipeline_runner import (
    get_status, start_pipeline, get_cached
)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/run")
def run_pipeline(dataset: str = "synthetic", sweep: bool = True):
    """Trigger the full ML pipeline in the background."""
    result = start_pipeline(dataset=dataset, run_sweep=sweep)
    return result


@router.get("/status")
def pipeline_status():
    """Get current pipeline execution status."""
    return get_status()


@router.get("/results")
def all_results():
    """Return all cached pipeline results."""
    return {
        "audit":      get_cached("audit"),
        "mitigation": get_cached("mitigation"),
        "fairness":   get_cached("fairness"),
        "tar_far":    get_cached("tar_far"),
        "plots":      get_cached("plots"),
    }
