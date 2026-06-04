"""
Module 2: Mitigation Pipeline

Orchestrates both mitigation techniques:
  1. Adversarial attribute disentanglement (λ-sweep)
  2. Gaussian noise injection (σ-sweep)

Produces before/after audit comparison and trade-off curve data.
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import (
    RANDOM_SEED, TEST_SIZE, RESULTS_DIR, DATASET_NAME,
    ADV_LAMBDA, ATTRIBUTES
)
from ml_engine.data.generate_synthetic_celeba import load_dataset
from ml_engine.module1_audit.audit_pipeline import run_audit_pipeline, _save_json
from ml_engine.module1_audit.attacker import run_attribute_audit
from ml_engine.module2_mitigate.adversarial import (
    train_adversarial_disentangler, apply_disentangler, run_lambda_sweep
)
from ml_engine.module2_mitigate.noise_injection import (
    inject_gaussian_noise, run_noise_sweep
)


def _quick_audit(X_train, X_test, y_train_dict, y_test_dict, tag=""):
    """Lightweight audit (LR only for speed in sweep)."""
    return run_attribute_audit(
        X_train, X_test, y_train_dict, y_test_dict, tag=tag
    )


def run_mitigation_pipeline(
    dataset_name: str = DATASET_NAME,
    run_sweep: bool = True,
    save: bool = True,
) -> dict:
    """
    Full mitigation pipeline.

    Returns
    -------
    dict containing:
      - baseline_results  : pre-mitigation audit
      - adv_results       : post-adversarial audit (best λ)
      - noise_results     : post-noise audit (best σ)
      - adv_sweep         : λ-sweep trade-off data
      - noise_sweep       : σ-sweep trade-off data
      - comparison        : side-by-side before/after table
    """
    print(f"\n{'='*60}")
    print(f"  MODULE 2: ATTRIBUTE LEAKAGE MITIGATION")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    ds = load_dataset(dataset_name)
    embeddings = ds["embeddings"]
    labels = {k: ds[k] for k in ["gender", "age_group", "ethnicity"]}

    idx_all = np.arange(len(embeddings))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=labels["gender"]
    )

    X_train = embeddings[idx_train]
    X_test  = embeddings[idx_test]
    y_train_dict = {attr: labels[attr][idx_train] for attr in ATTRIBUTES}
    y_test_dict  = {attr: labels[attr][idx_test]  for attr in ATTRIBUTES}

    # ── Baseline audit ────────────────────────────────────────────────────────
    print("\n[Mitigation] Running BASELINE audit …")
    baseline = run_audit_pipeline(
        embeddings=embeddings, labels=labels, tag="baseline", save=save
    )

    # ── Adversarial disentanglement ───────────────────────────────────────────
    print("\n[Mitigation] Training adversarial disentangler (λ={:.2f}) …".format(ADV_LAMBDA))
    adv_model, adv_history = train_adversarial_disentangler(
        X_train, y_train_dict, lam=ADV_LAMBDA
    )
    X_train_adv = apply_disentangler(adv_model, X_train)
    X_test_adv  = apply_disentangler(adv_model, X_test)

    adv_audit = run_attribute_audit(
        X_train_adv, X_test_adv, y_train_dict, y_test_dict, tag="post_adversarial"
    )

    # ── Noise injection (best σ from grid = 0.1) ──────────────────────────────
    best_sigma = 0.1
    X_train_noise = inject_gaussian_noise(X_train, best_sigma)
    X_test_noise  = inject_gaussian_noise(X_test,  best_sigma)

    noise_audit = run_attribute_audit(
        X_train_noise, X_test_noise, y_train_dict, y_test_dict, tag="post_noise"
    )

    # ── Sweeps ────────────────────────────────────────────────────────────────
    adv_sweep   = []
    noise_sweep = []
    if run_sweep:
        adv_sweep = run_lambda_sweep(
            X_train, X_test, y_train_dict, y_test_dict,
            audit_fn=_quick_audit
        )
        noise_sweep = run_noise_sweep(
            X_train, X_test, y_train_dict, y_test_dict,
            audit_fn=_quick_audit
        )

    # ── Build comparison table ─────────────────────────────────────────────────
    comparison = _build_comparison(baseline, adv_audit, noise_audit)

    output = {
        "baseline_results":  _strip_large_fields(baseline),
        "adv_results":       _strip_large_fields(adv_audit),
        "noise_results":     _strip_large_fields(noise_audit),
        "adv_sweep":         adv_sweep,
        "noise_sweep":       noise_sweep,
        "comparison":        comparison,
        "adv_training_history": adv_history,
    }

    if save:
        out = RESULTS_DIR / "mitigation_results.json"
        _save_json(output, out)
        print(f"\n[Mitigation] Results saved to {out}")

    # Print summary
    print(f"\n[Mitigation] ── Comparison Summary ─────────────────────")
    for row in comparison:
        print(
            f"  {row['attribute']:12s} │ "
            f"AUC baseline={row['baseline_auc']:.3f} → "
            f"adv={row['adv_auc']:.3f} (↓{row['adv_reduction_pct']:.1f}%) │ "
            f"noise={row['noise_auc']:.3f} (↓{row['noise_reduction_pct']:.1f}%)"
        )

    return output


def _build_comparison(baseline, adv, noise) -> list:
    rows = []
    for attr in ATTRIBUTES:
        if attr not in baseline:
            continue
        b_auc = baseline[attr]["best_attacker_auc"]
        a_auc = adv[attr]["best_attacker_auc"]
        n_auc = noise[attr]["best_attacker_auc"]

        rows.append({
            "attribute":           attr,
            "baseline_auc":        round(b_auc, 4),
            "adv_auc":             round(a_auc, 4),
            "noise_auc":           round(n_auc, 4),
            "adv_reduction":       round(b_auc - a_auc, 4),
            "noise_reduction":     round(b_auc - n_auc, 4),
            "adv_reduction_pct":   round((b_auc - a_auc) / (b_auc + 1e-8) * 100, 2),
            "noise_reduction_pct": round((b_auc - n_auc) / (b_auc + 1e-8) * 100, 2),
            "baseline_risk":       baseline[attr]["risk_level"],
            "adv_risk":            adv[attr]["risk_level"],
            "noise_risk":          noise[attr]["risk_level"],
        })
    return rows


def _strip_large_fields(d: dict) -> dict:
    """Remove verbose classification_report for storage efficiency."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if kk != "classification_report"}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    run_mitigation_pipeline()
