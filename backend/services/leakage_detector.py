"""
Leakage Detector Service

Thin wrapper that delegates attribute leakage detection to the ML engine.
Returns a structured result compatible with the pipeline's audit format.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def detect_leakage(dataset_name: str = "synthetic") -> dict:
    """
    Run a quick attribute leakage detection pass using the ML engine.

    Parameters
    ----------
    dataset_name : 'synthetic' | 'uploaded'

    Returns
    -------
    dict with leakage_score, risk_level, and per-attribute breakdown
    """
    from ml_engine.data.generate_synthetic_celeba import load_dataset
    from ml_engine.module1_audit.audit_pipeline import run_audit_pipeline
    import numpy as np

    ds = load_dataset(dataset_name)
    results = run_audit_pipeline(
        embeddings=ds["embeddings"],
        labels={k: ds[k] for k in ["gender", "age_group", "ethnicity"]},
        tag="quick_detect",
        save=False,
    )

    summary = results.get("__summary__", {})
    return {
        "leakage_score": round(float(summary.get("mean_leakage_score", 0)), 4),
        "max_auc":       round(float(summary.get("max_attacker_auc", 0.5)), 4),
        "risk_level":    summary.get("overall_risk_level", "UNKNOWN"),
        "per_attribute": {
            attr: {
                "auc":          round(results[attr]["best_attacker_auc"], 4),
                "leakage":      round(results[attr]["best_leakage_score"], 4),
                "risk":         results[attr]["risk_level"],
            }
            for attr in ["gender", "age_group", "ethnicity"]
            if attr in results
        },
    }
