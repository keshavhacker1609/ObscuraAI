"""
Module 1: Attribute Leakage Auditing Pipeline

Full orchestration of:  dataset loading → train/test split → attacker
training → metrics computation → result persistence.

References:
  Morales et al. (2020) "SensitiveNets", IEEE TPAMI.
  Terhorst et al. (2021) "Comprehensive Study of Face Recognition Biases", IEEE TIFS.
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import (
    RANDOM_SEED, TEST_SIZE, RESULTS_DIR, ATTRIBUTES, DATASET_NAME
)
from ml_engine.data.generate_synthetic_celeba import load_dataset
from ml_engine.module1_audit.attacker import run_attribute_audit


def run_audit_pipeline(
    dataset_name: str = DATASET_NAME,
    embeddings: np.ndarray = None,
    labels: dict = None,
    tag: str = "baseline",
    save: bool = True,
) -> dict:
    """
    End-to-end auditing pipeline.

    Parameters
    ----------
    dataset_name : which dataset to load if embeddings not provided
    embeddings   : pre-computed embeddings (N, D), optional
    labels       : dict of attribute label arrays, optional
    tag          : identifier for this audit run (e.g. "baseline" / "post_adv")
    save         : persist JSON results to disk

    Returns
    -------
    Structured audit results dict
    """
    print(f"\n{'='*60}")
    print(f"  MODULE 1: ATTRIBUTE LEAKAGE AUDITING  [tag={tag}]")
    print(f"{'='*60}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if embeddings is None or labels is None:
        print(f"\n[Audit] Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name)
        embeddings = ds["embeddings"]
        labels = {
            "gender":    ds["gender"],
            "age_group": ds["age_group"],
            "ethnicity": ds["ethnicity"],
        }

    N, D = embeddings.shape
    print(f"[Audit] Dataset: N={N:,} samples, D={D}-dim embeddings")

    # ── 2. Train / test split (stratified on gender) ──────────────────────────
    idx_all = np.arange(N)
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels["gender"],
    )

    X_train = embeddings[idx_train]
    X_test  = embeddings[idx_test]
    y_train_dict = {attr: labels[attr][idx_train] for attr in ATTRIBUTES}
    y_test_dict  = {attr: labels[attr][idx_test]  for attr in ATTRIBUTES}

    print(f"[Audit] Split: Train={len(X_train):,}  Test={len(X_test):,}")

    # ── 3. Run attackers ──────────────────────────────────────────────────────
    audit_results = run_attribute_audit(
        X_train, X_test, y_train_dict, y_test_dict,
        embed_dim=D, tag=tag
    )

    # ── 4. Compute summary ────────────────────────────────────────────────────
    summary = _build_summary(audit_results)
    audit_results["__summary__"] = summary

    print(f"\n[Audit] ── Summary ──────────────────────────────────────")
    print(f"  Mean Leakage Score : {summary['mean_leakage_score']:.3f}")
    print(f"  Overall Risk Level : {summary['overall_risk_level']}")
    for attr, data in audit_results.items():
        if attr.startswith("__"):
            continue
        print(f"  {attr:12s} → AUC={data['best_attacker_auc']:.3f}  Risk={data['risk_level']}")

    # ── 5. Persist ────────────────────────────────────────────────────────────
    if save:
        out = RESULTS_DIR / f"audit_{tag}.json"
        _save_json(audit_results, out)
        print(f"\n[Audit] Results saved to {out}")

    return audit_results


def _build_summary(results: dict) -> dict:
    from ml_engine.config import RISK_THRESHOLDS

    per_attr = {k: v for k, v in results.items() if not k.startswith("__")}
    mean_leak = float(np.mean([v["best_leakage_score"] for v in per_attr.values()]))
    max_auc   = float(np.max([v["best_attacker_auc"]   for v in per_attr.values()]))

    overall_risk = "NEGLIGIBLE"
    for level, thresh in RISK_THRESHOLDS.items():
        if max_auc >= thresh:
            overall_risk = level
            break

    return {
        "mean_leakage_score":   mean_leak,
        "max_attacker_auc":     max_auc,
        "overall_risk_level":   overall_risk,
        "n_attributes_audited": len(per_attr),
    }


def _save_json(data: dict, path: Path):
    """Recursively convert numpy types for JSON serialisation."""
    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj

    def _recurse(obj):
        if isinstance(obj, dict):  return {k: _recurse(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_recurse(i) for i in obj]
        return _convert(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_recurse(data), f, indent=2)


if __name__ == "__main__":
    run_audit_pipeline()
