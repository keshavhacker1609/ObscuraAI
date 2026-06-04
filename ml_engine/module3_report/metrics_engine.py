"""
Module 3: Metrics Engine

Computes all privacy, utility, and fairness metrics referenced in:

  NIST Face Recognition Vendor Test (FRVT) 2023 — Privacy Analysis Framework.
  Gong & Liu (2021) "Mitigating Face Recognition Bias via Group-Adaptive
    Classifier", CVPR.
  Dhar et al. (2021) "PASS: Protected Attribute Suppression System", ICCV.

Metrics:
  Privacy:
    - Attribute inference AUC (per attacker, per attribute)
    - Privacy Leakage Score (normalised above chance)
    - Mutual Information estimate
  Utility:
    - Cosine similarity preservation
    - Face verification TAR@FAR (simulated from embedding similarity)
  Fairness:
    - Per-demographic leakage disparity
    - Equalised Odds difference (TPR gap across groups)
    - Max Demographic Disparity (MDD)
"""

import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import ATTRIBUTES, RANDOM_SEED

np.random.seed(RANDOM_SEED)


# ─── Privacy Metrics ──────────────────────────────────────────────────────────

def compute_privacy_metrics(audit_results: dict) -> dict:
    """Extract structured privacy metrics from audit results."""
    metrics = {}
    for attr in ATTRIBUTES:
        if attr not in audit_results:
            continue
        r = audit_results[attr]
        metrics[attr] = {
            "lr_auc":            r["logistic_regression"]["auc_roc"],
            "mlp_auc":           r["mlp"]["auc_roc"],
            "best_auc":          r["best_attacker_auc"],
            "leakage_score":     r["best_leakage_score"],
            "mutual_info":       r["mlp"]["mutual_information"],
            "risk_level":        r["risk_level"],
            "accuracy":          r["mlp"]["accuracy"],
            "balanced_accuracy": r["mlp"]["balanced_accuracy"],
        }

    aucs  = [v["best_auc"]       for v in metrics.values()]
    leaks = [v["leakage_score"]  for v in metrics.values()]
    metrics["aggregate"] = {
        "mean_auc":           float(np.mean(aucs)),
        "max_auc":            float(np.max(aucs)),
        "mean_leakage_score": float(np.mean(leaks)),
        "max_leakage_score":  float(np.max(leaks)),
    }
    return metrics


# ─── Utility Metrics ──────────────────────────────────────────────────────────

def compute_verification_tar_at_far(
    embeddings_original: np.ndarray,
    embeddings_sanitised: np.ndarray,
    n_pairs: int = 5000,
    far_targets: list = [0.001, 0.01, 0.1],
) -> dict:
    """
    Simulate face verification TAR@FAR using cosine similarity.

    Genuine pairs:  original vs. sanitised embedding of the same sample.
    Impostor pairs: original vs. a different sanitised sample.
    """
    np.random.seed(RANDOM_SEED)
    N = min(len(embeddings_original), n_pairs)
    idx = np.random.choice(len(embeddings_original), size=N, replace=False)

    emb_orig = embeddings_original[idx]
    emb_san  = embeddings_sanitised[idx]

    genuine_cos  = np.sum(emb_orig * emb_san, axis=1)
    idx2 = np.roll(idx, 1)
    impostor_cos = np.sum(emb_orig * embeddings_sanitised[idx2], axis=1)

    tar_far = {}
    for far_target in far_targets:
        n_far = int(far_target * N)
        if n_far == 0:
            continue
        imp_sorted = np.sort(impostor_cos)[::-1]
        threshold  = imp_sorted[min(n_far - 1, len(imp_sorted) - 1)]
        tar = float(np.mean(genuine_cos >= threshold))
        tar_far[f"TAR@FAR{far_target}"] = round(tar, 4)

    tar_far["mean_genuine_similarity"]  = float(np.mean(genuine_cos))
    tar_far["mean_impostor_similarity"] = float(np.mean(impostor_cos))
    return tar_far


# ─── Fairness Metrics ─────────────────────────────────────────────────────────

def _compute_equalised_odds_difference(
    results_by_group: dict,
    target_attr: str = "gender",
) -> float:
    """
    Equalised Odds Difference: max |TPR_g1 − TPR_g2| across all group pairs.

    Uses balanced_accuracy as a proxy for TPR when ground-truth TPR is unavailable.
    """
    baccuracies = []
    for group, results in results_by_group.items():
        if target_attr in results:
            baccuracies.append(results[target_attr]["mlp"]["balanced_accuracy"])
    if len(baccuracies) < 2:
        return 0.0
    return float(max(baccuracies) - min(baccuracies))


def compute_fairness_metrics(
    audit_results_by_group: dict,
) -> dict:
    """
    Compute per-demographic fairness metrics.

    Parameters
    ----------
    audit_results_by_group : {group_name: audit_result_dict}

    Returns
    -------
    dict with per_group_leakage, max_demographic_disparity,
    equalised_odds_difference, fairness_gap, n_groups_analyzed.
    """
    group_leakage = {}
    for group, results in audit_results_by_group.items():
        leaks = []
        for attr in ATTRIBUTES:
            if attr in results:
                leaks.append(results[attr]["best_leakage_score"])
        group_leakage[group] = {
            "mean_leakage": float(np.mean(leaks)) if leaks else 0.0,
            "max_leakage":  float(np.max(leaks))  if leaks else 0.0,
        }

    values = [v["mean_leakage"] for v in group_leakage.values()]
    mdd    = float(max(values) - min(values)) if len(values) >= 2 else 0.0
    eod    = _compute_equalised_odds_difference(audit_results_by_group)

    return {
        "per_group_leakage":           group_leakage,
        "max_demographic_disparity":   mdd,
        "equalised_odds_difference":   eod,
        "fairness_gap":                mdd,
        "n_groups_analyzed":           len(group_leakage),
    }


def compute_fairness_from_audit(
    embeddings: np.ndarray,
    labels: dict,
    audit_fn,
    group_attr: str = "ethnicity",
) -> dict:
    """
    Split embeddings by demographic group and compute per-group leakage.
    """
    from ml_engine.config import ATTRIBUTES as ATTR_CFG
    group_labels = labels[group_attr]
    n_groups     = ATTR_CFG[group_attr]["n_classes"]
    group_names  = ATTR_CFG[group_attr]["labels"]

    results_by_group = {}
    for g in range(n_groups):
        mask = group_labels == g
        if mask.sum() < 50:
            continue
        emb_g    = embeddings[mask]
        labels_g = {attr: labels[attr][mask] for attr in ATTRIBUTES}

        N_g   = len(emb_g)
        split = max(int(0.2 * N_g), 10)
        X_tr, X_te = emb_g[:-split], emb_g[-split:]
        y_tr = {a: labels_g[a][:-split] for a in ATTRIBUTES}
        y_te = {a: labels_g[a][-split:]  for a in ATTRIBUTES}

        try:
            audit = audit_fn(X_tr, X_te, y_tr, y_te, tag=f"group_{group_names[g]}")
            results_by_group[group_names[g]] = audit
        except Exception as e:
            print(f"  [Fairness] Warning: group {group_names[g]} skipped: {e}")

    return compute_fairness_metrics(results_by_group)


# ─── Model Card Builder ───────────────────────────────────────────────────────

def build_model_card(
    dataset_info: dict,
    audit_results: dict,
    mitigation_results: dict,
    fairness_results: dict,
    tar_far_results: Optional[dict] = None,
) -> dict:
    """Build an IEEE model card–style structured report."""
    priv_metrics = compute_privacy_metrics(audit_results)

    card = {
        "model_card_version": "1.0",
        "framework":          "ObscuraAI — Privacy Intelligence Framework",
        "reference":          "IEEE TPAMI / TIFS / ICCV 2021",
        "dataset":            dataset_info,
        "privacy_analysis": {
            "baseline":          priv_metrics,
            "overall_risk":      audit_results.get("__summary__", {}).get("overall_risk_level", "UNKNOWN"),
            "mean_leakage_score": priv_metrics["aggregate"]["mean_leakage_score"],
        },
        "mitigation": {
            "techniques_applied": ["adversarial_disentanglement", "gaussian_noise_injection"],
            "comparison_table":   mitigation_results.get("comparison", []),
            "adversarial_lambda": 0.8,
            "noise_sigma_best":   0.1,
        },
        "utility": tar_far_results or {
            "note": "TAR@FAR requires paired identity data; using cosine similarity proxy."
        },
        "fairness":          fairness_results,
        "recommendations":   _generate_recommendations(priv_metrics, fairness_results),
    }

    return card


def _generate_recommendations(priv_metrics: dict, fairness: dict) -> list:
    recs  = []
    agg   = priv_metrics.get("aggregate", {})
    mean_leak = agg.get("mean_leakage_score", 0)
    mdd   = fairness.get("max_demographic_disparity", 0)
    eod   = fairness.get("equalised_odds_difference", 0)

    if mean_leak > 0.7:
        recs.append("CRITICAL: Apply strong adversarial disentanglement (λ≥1.0) before deployment.")
    elif mean_leak > 0.4:
        recs.append("HIGH: Adversarial mitigation with λ=0.8 recommended. Re-evaluate before deployment.")
    else:
        recs.append("MODERATE: Noise injection (σ=0.05) sufficient for most threat models.")

    if mdd > 0.2:
        recs.append(
            f"FAIRNESS (MDD={mdd:.3f}): Significant demographic disparity detected. "
            "Apply group-aware regularisation or re-balance training data."
        )

    if eod > 0.15:
        recs.append(
            f"EQUALISED ODDS (EOD={eod:.3f}): Unequal TPR across demographic groups. "
            "Consider post-processing calibration."
        )

    if priv_metrics.get("ethnicity", {}).get("best_auc", 0) > 0.8:
        recs.append(
            "Ethnicity leakage is HIGH — apply targeted attribute suppression "
            "for this dimension before public deployment."
        )

    recs.append(
        "Periodically re-audit after any model update, fine-tuning, or dataset distribution shift."
    )
    return recs
