"""
Pipeline Runner Service

Manages background execution of the ML pipeline with live status tracking.
Results are cached in memory and persisted to disk via the ML engine.
Each pipeline step emits a tamper-evident audit log entry.
"""

import json
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml_engine.config import RESULTS_DIR, PLOTS_DIR
from backend.services.audit_log import add_entry as _log

# ─── Global State ─────────────────────────────────────────────────────────────

_pipeline_state = {
    "status":       "idle",
    "progress":     0.0,
    "message":      "Ready to run",
    "started_at":   None,
    "completed_at": None,
    "error":        None,
    "dataset":      None,
    "run_id":       None,
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

    plots = {}
    if PLOTS_DIR.exists():
        for p in PLOTS_DIR.glob("*.png"):
            plots[p.stem] = p.name
    set_cached("plots", plots)

    if get_cached("audit"):
        _set_state(status="completed", message="Loaded previous results from disk.")
        _log(
            "Prior results loaded from disk",
            module="INIT",
            level="INFO",
            metadata={"plots": str(len(plots))},
        )


# ─── Background Pipeline ──────────────────────────────────────────────────────

def _run_pipeline_thread(dataset: str, run_sweep: bool, run_id: str):
    """Background thread target for the full ML pipeline."""
    try:
        _set_state(
            status="running", progress=5.0,
            message="Generating / loading dataset …",
            started_at=datetime.now(timezone.utc).isoformat(),
            error=None,
            dataset=dataset,
            run_id=run_id,
        )
        _log("Pipeline started", module="PIPELINE", level="INFO",
             metadata={"run_id": run_id, "dataset": dataset, "sweep": str(run_sweep)})

        # Lazy imports to avoid slow startup
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

        # ── Step 0: Dataset ───────────────────────────────────────────────────
        _set_state(progress=8.0, message="Loading dataset …")
        _log(f"Loading dataset: {dataset}", module="DATA", level="INFO")
        ds = load_dataset(dataset)
        embeddings = ds["embeddings"]
        labels = {k: ds[k] for k in ["gender", "age_group", "ethnicity"]}
        _log(
            f"Dataset loaded: N={len(embeddings):,} × D={embeddings.shape[1]}",
            module="DATA", level="INFO",
            metadata={"n_samples": str(len(embeddings)), "embed_dim": str(embeddings.shape[1])},
        )

        # ── Step 1: Module 1 — Audit ──────────────────────────────────────────
        _set_state(progress=15.0, message="Module 1: Running attribute leakage audit …")
        _log("Module 1 started: Attribute Leakage Auditing", module="AUDIT", level="INFO")
        audit = run_audit_pipeline(
            embeddings=embeddings, labels=labels, tag="baseline", save=True
        )
        set_cached("audit", audit)
        summary = audit.get("__summary__", {})
        _log(
            f"Module 1 complete — overall risk: {summary.get('overall_risk_level', 'N/A')} "
            f"| mean leakage: {summary.get('mean_leakage_score', 0):.4f} "
            f"| max AUC: {summary.get('max_attacker_auc', 0):.4f}",
            module="AUDIT", level="INFO",
            metadata={
                "overall_risk":   summary.get("overall_risk_level", "N/A"),
                "mean_leakage":   f"{summary.get('mean_leakage_score', 0):.4f}",
                "max_auc":        f"{summary.get('max_attacker_auc', 0):.4f}",
                "n_attrs":        str(summary.get("n_attributes_audited", 0)),
            },
        )

        # ── Step 2: Module 2 — Mitigation ────────────────────────────────────
        _set_state(progress=35.0, message="Module 2: Running mitigation pipeline …")
        _log("Module 2 started: Adversarial Disentanglement + Noise Injection",
             module="MITIGATE", level="INFO")
        mitigation = run_mitigation_pipeline(
            dataset_name=dataset, run_sweep=run_sweep, save=True
        )
        set_cached("mitigation", mitigation)
        comparison = mitigation.get("comparison", [])
        if comparison:
            avg_adv = sum(r["adv_reduction_pct"] for r in comparison) / len(comparison)
            avg_noise = sum(r["noise_reduction_pct"] for r in comparison) / len(comparison)
            _log(
                f"Module 2 complete — avg AUC reduction: adv={avg_adv:.1f}%, noise={avg_noise:.1f}%",
                module="MITIGATE", level="INFO",
                metadata={"avg_adv_reduction": f"{avg_adv:.1f}%", "avg_noise_reduction": f"{avg_noise:.1f}%"},
            )

        # ── Step 3a: Fairness ─────────────────────────────────────────────────
        _set_state(progress=65.0, message="Module 3a: Computing fairness metrics …")
        _log("Computing per-demographic fairness metrics (MDD, equalized odds)",
             module="FAIRNESS", level="INFO")

        def _audit_fn(X_tr, X_te, y_tr, y_te, tag=""):
            return run_attribute_audit(X_tr, X_te, y_tr, y_te, tag=tag)

        fairness = compute_fairness_from_audit(embeddings, labels, _audit_fn)
        set_cached("fairness", fairness)
        mdd = fairness.get("max_demographic_disparity", 0)
        _log(
            f"Fairness analysis complete — MDD={mdd:.4f} | groups={fairness.get('n_groups_analyzed', 0)}",
            module="FAIRNESS", level="INFO",
            metadata={"mdd": f"{mdd:.4f}", "n_groups": str(fairness.get("n_groups_analyzed", 0))},
        )

        # ── Step 3b: TAR@FAR ──────────────────────────────────────────────────
        _set_state(progress=72.0, message="Module 3b: Computing TAR@FAR utility …")
        _log("Computing face verification utility (TAR@FAR)", module="UTILITY", level="INFO")
        idx = np.arange(len(embeddings))
        idx_tr, idx_te = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=labels["gender"]
        )
        y_tr_dict = {a: labels[a][idx_tr] for a in ATTRIBUTES}
        adv_model, _ = train_adversarial_disentangler(
            embeddings[idx_tr], y_tr_dict, lam=ADV_LAMBDA, epochs=20, verbose=False
        )
        emb_san = apply_disentangler(adv_model, embeddings[idx_te])
        tar_far = compute_verification_tar_at_far(embeddings[idx_te], emb_san)
        set_cached("tar_far", tar_far)
        _log(
            f"TAR@FAR computed — genuine similarity={tar_far.get('mean_genuine_similarity', 0):.4f}",
            module="UTILITY", level="INFO",
            metadata={k: f"{v:.4f}" for k, v in tar_far.items() if isinstance(v, float)},
        )

        # ── Step 3c: Visualizations ───────────────────────────────────────────
        _set_state(progress=82.0, message="Module 3c: Generating 7 publication-quality plots …")
        _log("Generating IEEE-publication-quality visualizations", module="VIZ", level="INFO")
        audit_post = mitigation.get("adv_results", {})
        plot_paths = generate_all_plots(
            audit_baseline=audit,
            audit_post_adv=audit_post,
            mitigation_results=mitigation,
            fairness_results=fairness,
        )
        set_cached("plots", {k: Path(v).name for k, v in plot_paths.items() if v})
        _log(
            f"{len(plot_paths)} plots generated",
            module="VIZ", level="INFO",
            metadata={"plots": ", ".join(plot_paths.keys())},
        )

        # ── Step 3d: Model Card ───────────────────────────────────────────────
        _set_state(progress=93.0, message="Generating IEEE model card with SHA-256 signatures …")
        _log("Generating IEEE model card report with cryptographic hash", module="REPORT", level="INFO")
        report = generate_full_report(
            audit_baseline=audit,
            mitigation_results=mitigation,
            fairness_results=fairness,
            plot_paths=plot_paths,
            save=True,
        )
        set_cached("model_card", report)
        _log(
            f"Model card generated — audit hash: {report.get('audit_hash', 'N/A')[:16]}...",
            module="REPORT", level="INFO",
            metadata={
                "audit_hash":   report.get("audit_hash", "N/A")[:32],
                "generated_at": report.get("generated_at", "N/A"),
            },
        )

        _set_state(
            status="completed", progress=100.0,
            message="Pipeline completed successfully.",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _log(
            f"Pipeline run {run_id} COMPLETED successfully",
            module="PIPELINE", level="INFO",
            metadata={"run_id": run_id, "risk": summary.get("overall_risk_level", "N/A")},
        )

    except Exception as exc:
        tb = traceback.format_exc()
        _set_state(
            status="error", progress=0.0,
            message=f"Pipeline failed: {exc}",
            error=tb,
        )
        _log(
            f"Pipeline run {run_id} FAILED: {exc}",
            module="PIPELINE", level="ERROR",
            metadata={"run_id": run_id, "error": str(exc)[:300]},
        )
        print(f"[PipelineRunner] ERROR:\n{tb}")


def start_pipeline(dataset: str = "synthetic", run_sweep: bool = True) -> dict:
    """Start the pipeline in a background thread."""
    import uuid
    with _lock:
        if _pipeline_state["status"] == "running":
            return {"started": False, "reason": "Pipeline already running"}

    run_id = uuid.uuid4().hex[:12].upper()
    _log(
        f"Pipeline run {run_id} triggered",
        module="PIPELINE", level="INFO",
        metadata={"run_id": run_id, "dataset": dataset, "sweep": str(run_sweep)},
    )

    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(dataset, run_sweep, run_id),
        daemon=True,
    )
    t.start()
    return {"started": True, "run_id": run_id, "message": "Pipeline started in background"}


# ─── Init ─────────────────────────────────────────────────────────────────────
load_results_from_disk()
