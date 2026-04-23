"""
AI Privacy Intelligence — FastAPI Backend

Main application entry point.
"""

import sys
from pathlib import Path

# Ensure project root is on path so ml_engine imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes import audit, upload
from routes.pipeline       import router as pipeline_router
from routes.mitigate       import router as mitigate_router
from routes.report         import router as report_router
from routes.metrics        import router as metrics_router
from routes.visualizations import router as viz_router
from ml_engine.config import STATIC_DIR, PLOTS_DIR

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Privacy Intelligence API",
    description=(
        "Unified framework for auditing, mitigating, and reporting "
        "attribute leakage in large-scale face recognition systems."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production: restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static files (plots) ──────────────────────────────────────────────────────
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(audit.router)
app.include_router(upload.router)
app.include_router(pipeline_router)
app.include_router(mitigate_router)
app.include_router(report_router)
app.include_router(metrics_router)
app.include_router(viz_router)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "name":    "AI Privacy Intelligence API",
        "version": "2.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    from backend.services.pipeline_runner import get_status
    ps = get_status()
    return {
        "api":      "healthy",
        "pipeline": ps["status"],
        "progress": ps["progress"],
    }
