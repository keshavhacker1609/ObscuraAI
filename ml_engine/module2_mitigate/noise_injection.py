"""
Module 2: Noise Injection Mitigation

Controlled Gaussian noise injection at multiple σ levels as a simple
but well-studied baseline for attribute leakage mitigation.

References:
  Biggio et al. (2012) "Poisoning Attacks against SVMs", ICML.
  Dwork et al. (2014) "The Algorithmic Foundations of Differential Privacy".
  Mirjalili et al. (2020) "PrivacyNet: Semi-Adversarial Networks for
    Attribute Obfuscation", IEEE TIP.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import NOISE_SIGMA_GRID, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def inject_gaussian_noise(
    embeddings: np.ndarray,
    sigma: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Add isotropic Gaussian noise to embeddings.

    Parameters
    ----------
    embeddings : (N, D) face embeddings (assumed L2-normalised)
    sigma      : noise standard deviation
    normalize  : re-normalise after injection (recommended for face embeddings)

    Returns
    -------
    Noisy embeddings (N, D)
    """
    noise = np.random.normal(0.0, sigma, size=embeddings.shape).astype(np.float32)
    noisy = embeddings + noise
    if normalize:
        norms = np.linalg.norm(noisy, axis=1, keepdims=True)
        noisy = noisy / (norms + 1e-8)
    return noisy


def compute_embedding_drift(
    original: np.ndarray,
    sanitised: np.ndarray,
) -> dict:
    """
    Measure the distortion introduced by noise injection.

    Returns
    -------
    dict with:
      - mean_cosine_similarity  : identity utility proxy
      - mean_l2_distance        : raw distortion
      - snr_db                  : signal-to-noise ratio in dB
    """
    # Cosine similarity
    cos_sims = np.sum(original * sanitised, axis=1)   # already L2-normalised
    mean_cos = float(np.mean(cos_sims))

    # L2 distance
    l2_dist = np.linalg.norm(original - sanitised, axis=1)
    mean_l2 = float(np.mean(l2_dist))

    # SNR
    signal_power = float(np.mean(np.sum(original**2, axis=1)))
    noise_power  = float(np.mean(np.sum((sanitised - original)**2, axis=1)))
    snr_db = 10.0 * np.log10(signal_power / (noise_power + 1e-12))

    return {
        "mean_cosine_similarity": mean_cos,
        "mean_l2_distance":       mean_l2,
        "snr_db":                 float(snr_db),
    }


def run_noise_sweep(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train_dict: dict,
    y_test_dict:  dict,
    audit_fn,
    sigma_grid: list = NOISE_SIGMA_GRID,
) -> list:
    """
    Sweep noise levels and compute utility–privacy trade-off.

    Returns list of dicts with key metrics at each σ.
    """
    print(f"\n[NoiseInjection] Sweeping σ ∈ {sigma_grid} …")
    sweep_results = []

    for sigma in sigma_grid:
        print(f"\n  ── σ = {sigma:.3f} ───────────────────────────────")
        X_train_n = inject_gaussian_noise(X_train, sigma)
        X_test_n  = inject_gaussian_noise(X_test,  sigma)

        # Drift metrics
        drift = compute_embedding_drift(X_test, X_test_n)

        # Re-audit on noisy embeddings
        audit = audit_fn(X_train_n, X_test_n, y_train_dict, y_test_dict, tag=f"noise_s{sigma}")
        agg   = audit["__aggregate__"]

        sweep_results.append({
            "sigma":              sigma,
            "mean_leakage_score": agg["mean_leakage_score"],
            "max_leakage_score":  agg["max_leakage_score"],
            "identity_utility":   drift["mean_cosine_similarity"],
            "snr_db":             drift["snr_db"],
            "mean_l2_distance":   drift["mean_l2_distance"],
        })

        print(
            f"    → Leakage={agg['mean_leakage_score']:.3f}  "
            f"Utility={drift['mean_cosine_similarity']:.3f}  "
            f"SNR={drift['snr_db']:.1f}dB"
        )

    return sweep_results
