"""
ObscuraAI — FastAPI Backend
Main application entry point.
"""

import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows to prevent cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
from routes.logs           import router as logs_router
from ml_engine.config      import STATIC_DIR, PLOTS_DIR

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ObscuraAI — Privacy Intelligence API",
    description=(
        "Adversarial Demographic Leakage Auditing & Mitigation Framework. "
        "Detects, reduces, and documents sensitive demographic attribute leakage "
        "in face recognition embedding spaces with cryptographic audit trails."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static files (plots) ─────────────────────────────────────────────────────
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
app.include_router(logs_router)


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def _startup():
    from backend.services.audit_log import add_entry
    add_entry(
        "ObscuraAI API started",
        module="SYSTEM",
        level="INFO",
        metadata={"version": "2.1.0"},
    )


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "name":    "ObscuraAI Privacy Intelligence API",
        "version": "2.1.0",
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
