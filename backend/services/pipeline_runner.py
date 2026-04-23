"""
Pipeline Runner Service

Manages background execution of the ML pipeline with live status tracking.
Results are cached in memory and persisted to disk via the ML engine.
"""

import json
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml_engine.config import RESULTS_DIR, PLOTS_DIR

# ─── Global State ─────────────────────────────────────────────────────────────

_pipeline_state = {
    "status":       "idle",
    "progress":     0.0,
    "message":      "Ready to run",
    "started_at":   None,
    "completed_at": None,
    "error":        None,
}

_cached_results = {
    "audit":      None,
    "mitigation": None,
    "fairness":   None,
    "model_card": None,
    "plots":      {},
    "tar_far":    None,
}

_lock = threading.Lock()


# ─── State Accessors ──────────────────────────────────────────────────────────

def get_status() -> dict:
    with _lock:
        return dict(_pipeline_state)


def get_cached(key: str):
    with _lock:
        return _cached_results.get(key)


def set_cached(key: str, value):
    with _lock:
        _cached_results[key] = value


def _set_state(**kwargs):
    with _lock:
        _pipeline_state.update(kwargs)


# ─── Result Loaders (from disk) ───────────────────────────────────────────────

def _try_load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def load_results_from_disk():
    """Attempt to load prior results from disk on startup."""
    set_cached("audit",      _try_load_json(RESULTS_DIR / "audit_baseline.json"))
    set_cached("mitigation", _try_load_json(RESULTS_DIR / "mitigation_results.json"))
    set_cached("fairness",   _try_load_json(RESULTS_DIR / "fairness_results.json"))
    set_cached("model_card", _try_load_json(RESULTS_DIR / "model_card.json"))

    # Map plot names to filenames
    plots = {}
    if PLOTS_DIR.exists():
        for p in PLOTS_DIR.glob("*.png"):
            plots[p.stem] = p.name
    set_cached("plots", plots)

    if get_cached("audit"):
        _set_state(status="completed", message="Loaded previous results from disk.")


# ─── Background Pipeline ──────────────────────────────────────────────────────

def _run_pipeline_thread(dataset: str, run_sweep: bool):
    """Background thread target for the full ML pipeline."""
    try:
        _set_state(
            status="running", progress=5.0,
            message="Generating / loading dataset …",
            started_at=datetime.now(timezone.utc).isoformat(),
            error=None,
        )

        # Import lazily to avoid slow startup
        from ml_engine.data.generate_synthetic_celeba import load_dataset
        from ml_engine.module1_audit.audit_pipeline import run_audit_pipeline
        from ml_engine.module2_mitigate.mitigation_pipeline import run_mitigation_pipeline
        from ml_engine.module3_report.metrics_engine import (
            compute_fairness_from_audit, compute_verification_tar_at_far
        )
        from ml_engine.module3_report.visualizer import generate_all_plots
        from ml_engine.module3_report.report_generator import generate_full_report
        from ml_engine.module1_audit.attacker import run_attribute_audit
        from ml_engine.module2_mitigate.adversarial import (
            train_adversarial_disentangler, apply_disentangler
        )
        from ml_engine.config import ATTRIBUTES, ADV_LAMBDA
        from sklearn.model_selection import train_test_split
        import numpy as np

        ds = load_dataset(dataset)
        embeddings = ds["embeddings"]
        labels = {k: ds[k] for k in ["gender", "age_group", "ethnicity"]}

        # Module 1
        _set_state(progress=15.0, message="Module 1: Running attribute leakage audit …")
        audit = run_audit_pipeline(
            embeddings=embeddings, labels=labels, tag="baseline", save=True
        )
        set_cached("audit", audit)

        # Module 2
        _set_state(progress=35.0, message="Module 2: Running mitigation pipeline …")
        mitigation = run_mitigation_pipeline(
            dataset_name=dataset, run_sweep=run_sweep, save=True
        )
        set_cached("mitigation", mitigation)

        # Fairness
        _set_state(progress=65.0, message="Module 3a: Computing fairness metrics …")

        def _audit_fn(X_tr, X_te, y_tr, y_te, tag=""):
            return run_attribute_audit(X_tr, X_te, y_tr, y_te, tag=tag)

        fairness = compute_fairness_from_audit(embeddings, labels, _audit_fn)
        set_cached("fairness", fairness)

        # TAR@FAR
        _set_state(progress=72.0, message="Module 3b: Computing TAR@FAR utility …")
        idx = np.arange(len(embeddings))
        idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42,
                                          stratify=labels["gender"])
        y_tr_dict = {a: labels[a][idx_tr] for a in ATTRIBUTES}
        adv_model, _ = train_adversarial_disentangler(
            embeddings[idx_tr], y_tr_dict, lam=ADV_LAMBDA, epochs=20, verbose=False
        )
        emb_san = apply_disentangler(adv_model, embeddings[idx_te])
        tar_far = compute_verification_tar_at_far(embeddings[idx_te], emb_san)
        set_cached("tar_far", tar_far)

        # Visualizations
        _set_state(progress=82.0, message="Module 3c: Generating visualizations …")
        audit_post = mitigation.get("adv_results", {})
        plot_paths = generate_all_plots(
            audit_baseline=audit,
            audit_post_adv=audit_post,
            mitigation_results=mitigation,
            fairness_results=fairness,
        )
        set_cached("plots", {k: Path(v).name for k, v in plot_paths.items() if v})

        # Model card
        _set_state(progress=93.0, message="Generating model card report …")
        report = generate_full_report(
            audit_baseline=audit,
            mitigation_results=mitigation,
            fairness_results=fairness,
            plot_paths=plot_paths,
            save=True,
        )
        set_cached("model_card", report)

        _set_state(
            status="completed", progress=100.0,
            message="Pipeline completed successfully.",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as exc:
        tb = traceback.format_exc()
        _set_state(
            status="error", progress=0.0,
            message=f"Pipeline failed: {exc}",
            error=tb,
        )
        print(f"[PipelineRunner] ERROR:\n{tb}")


def start_pipeline(dataset: str = "synthetic", run_sweep: bool = True) -> dict:
    """Start the pipeline in a background thread."""
    with _lock:
        if _pipeline_state["status"] == "running":
            return {"started": False, "reason": "Pipeline already running"}

    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(dataset, run_sweep),
        daemon=True,
    )
    t.start()
    return {"started": True, "message": "Pipeline started in background"}


# ─── Init ─────────────────────────────────────────────────────────────────────
load_results_from_disk()
