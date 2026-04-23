"""
Master Pipeline Runner

Executes all 3 modules end-to-end:
  Module 1 → Attribute Leakage Auditing
  Module 2 → Leakage Mitigation (adversarial + noise)
  Module 3 → Reporting & Visualization

Usage:
  python run_full_pipeline.py [--dataset synthetic] [--skip-sweep]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml_engine.config import DATASET_NAME, RESULTS_DIR, ATTRIBUTES
from ml_engine.data.generate_synthetic_celeba import load_dataset, generate_synthetic_dataset
from ml_engine.module1_audit.audit_pipeline import run_audit_pipeline, _save_json
from ml_engine.module2_mitigate.mitigation_pipeline import run_mitigation_pipeline
from ml_engine.module3_report.metrics_engine import (
    compute_fairness_from_audit, compute_verification_tar_at_far
)
from ml_engine.module3_report.visualizer import generate_all_plots
from ml_engine.module3_report.report_generator import generate_full_report
from ml_engine.module1_audit.attacker import run_attribute_audit

import numpy as np
from sklearn.model_selection import train_test_split


def main(dataset_name: str = "synthetic", run_sweep: bool = True):
    t_start = time.time()

    print("\n" + "=" * 60)
    print("  AI PRIVACY INTELLIGENCE FRAMEWORK")
    print("  Full Pipeline Execution")
    print("=" * 60)

    # ── Generate / load dataset ────────────────────────────────────────────────
    print("\n[Pipeline] Step 0: Dataset preparation")
    ds = load_dataset(dataset_name)
    embeddings = ds["embeddings"]
    labels = {k: ds[k] for k in ["gender", "age_group", "ethnicity"]}
    print(f"  Loaded {len(embeddings):,} samples × {embeddings.shape[1]}-dim embeddings")

    # ── Module 1: Audit ────────────────────────────────────────────────────────
    print("\n[Pipeline] Step 1: Auditing …")
    audit_baseline = run_audit_pipeline(
        dataset_name=dataset_name,
        embeddings=embeddings,
        labels=labels,
        tag="baseline",
        save=True,
    )

    # ── Module 2: Mitigation ───────────────────────────────────────────────────
    print("\n[Pipeline] Step 2: Mitigation …")
    mitigation_results = run_mitigation_pipeline(
        dataset_name=dataset_name,
        run_sweep=run_sweep,
        save=True,
    )

    # ── Fairness analysis ─────────────────────────────────────────────────────
    print("\n[Pipeline] Step 3a: Fairness analysis …")

    def _audit_fn(X_tr, X_te, y_tr, y_te, tag=""):
        return run_attribute_audit(X_tr, X_te, y_tr, y_te, tag=tag)

    fairness_results = compute_fairness_from_audit(
        embeddings, labels, _audit_fn, group_attr="ethnicity"
    )
    _save_json(fairness_results, RESULTS_DIR / "fairness_results.json")
    print(f"  Max Demographic Disparity: {fairness_results['max_demographic_disparity']:.4f}")

    # ── TAR@FAR ───────────────────────────────────────────────────────────────
    print("\n[Pipeline] Step 3b: Utility (TAR@FAR) …")
    from ml_engine.module2_mitigate.adversarial import (
        train_adversarial_disentangler, apply_disentangler
    )
    from ml_engine.config import ADV_LAMBDA

    idx = np.arange(len(embeddings))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels["gender"])
    y_tr = {a: labels[a][idx_tr] for a in ATTRIBUTES}

    adv_model, _ = train_adversarial_disentangler(
        embeddings[idx_tr], y_tr, lam=ADV_LAMBDA, epochs=20, verbose=False
    )
    emb_san = apply_disentangler(adv_model, embeddings[idx_te])
    tar_far = compute_verification_tar_at_far(
        embeddings[idx_te], emb_san, n_pairs=2000
    )
    print(f"  TAR@FAR: {tar_far}")

    # ── Module 3: Visualizations ───────────────────────────────────────────────
    print("\n[Pipeline] Step 3c: Generating visualizations …")
    audit_post_adv = mitigation_results.get("adv_results", {})
    plot_paths = generate_all_plots(
        audit_baseline=audit_baseline,
        audit_post_adv=audit_post_adv,
        mitigation_results=mitigation_results,
        fairness_results=fairness_results,
    )

    # ── Module 3: Report ───────────────────────────────────────────────────────
    print("\n[Pipeline] Step 3d: Generating model card report …")
    report = generate_full_report(
        audit_baseline=audit_baseline,
        mitigation_results=mitigation_results,
        fairness_results=fairness_results,
        plot_paths=plot_paths,
        save=True,
    )

    # ── Done ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Results   → {RESULTS_DIR}")
    print(f"  Plots     → {RESULTS_DIR.parent / 'static' / 'plots'}")
    print(f"  Risk Level: {audit_baseline.get('__summary__', {}).get('overall_risk_level', 'N/A')}")
    print(f"{'═'*60}\n")

    return {
        "audit":      audit_baseline,
        "mitigation": mitigation_results,
        "fairness":   fairness_results,
        "tar_far":    tar_far,
        "plots":      plot_paths,
        "report":     report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Privacy Intelligence Full Pipeline")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic"], help="Dataset to use")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip λ/σ sweep (faster, less complete)")
    args = parser.parse_args()

    main(dataset_name=args.dataset, run_sweep=not args.skip_sweep)
