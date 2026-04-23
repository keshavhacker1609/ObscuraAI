"""Visualizations router — serve generated PNG plots."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from backend.services.pipeline_runner import get_cached
from ml_engine.config import PLOTS_DIR

router = APIRouter(prefix="/visualizations", tags=["Visualizations"])


@router.get("")
@router.get("/")
def list_visualizations():
    """List all available plot files."""
    plots = get_cached("plots") or {}

    # Also scan disk
    disk_plots = {}
    if PLOTS_DIR.exists():
        for p in PLOTS_DIR.glob("*.png"):
            disk_plots[p.stem] = p.name

    all_plots = {**disk_plots, **plots}
    return {"status": "ok", "plots": all_plots, "count": len(all_plots)}


@router.get("/{plot_name}")
def get_plot(plot_name: str):
    """Serve a specific plot PNG."""
    # Sanitize filename
    plot_name = plot_name.replace("..", "").strip("/\\")
    if not plot_name.endswith(".png"):
        plot_name += ".png"

    path = PLOTS_DIR / plot_name
    if not path.exists():
        raise HTTPException(404, f"Plot '{plot_name}' not found. Run the pipeline first.")

    return FileResponse(
        str(path),
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"},
    )
