"""
Global configuration for the AI Privacy Intelligence framework.
Covers dataset paths, model hyperparameters, attribute definitions,
and reproducibility seeds — aligned with IEEE experimental standards.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RESULTS_DIR     = ROOT_DIR / "results"
STATIC_DIR      = ROOT_DIR / "static"
PLOTS_DIR       = STATIC_DIR / "plots"
MODELS_DIR      = ROOT_DIR / "saved_models"

for _d in [DATA_DIR, RESULTS_DIR, STATIC_DIR, PLOTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED     = 42

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME    = "synthetic"          # "synthetic" | "celeba" | "fairface"
SYNTHETIC_N     = 12_000               # Number of synthetic samples
EMBEDDING_DIM   = 512                  # Face embedding dimension
TEST_SIZE       = 0.20
VAL_SIZE        = 0.10

# ─── Sensitive Attributes ─────────────────────────────────────────────────────
ATTRIBUTES = {
    "gender":    {"n_classes": 2, "labels": ["Female", "Male"]},
    "age_group": {"n_classes": 4, "labels": ["0-18", "19-35", "36-60", "60+"]},
    "ethnicity": {"n_classes": 5, "labels": ["White", "Black", "Asian", "Indian", "Other"]},
}

# ─── Attacker Models ──────────────────────────────────────────────────────────
LR_MAX_ITER     = 1000
MLP_HIDDEN      = (256, 128)
MLP_EPOCHS      = 50
MLP_LR          = 1e-3
MLP_BATCH_SIZE  = 256

# ─── Adversarial Mitigation ───────────────────────────────────────────────────
ADV_EPOCHS      = 60
ADV_LR          = 1e-4
ADV_LAMBDA      = 0.8               # Attribute adversary weight
ADV_LAMBDA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

# ─── Noise Injection ──────────────────────────────────────────────────────────
NOISE_SIGMA_GRID = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

# ─── Risk Thresholds (AUC-based) ──────────────────────────────────────────────
RISK_THRESHOLDS = {
    "CRITICAL": 0.90,
    "HIGH":     0.75,
    "MEDIUM":   0.60,
    "LOW":      0.55,   # ~random chance for binary
}

# ─── Plotting ─────────────────────────────────────────────────────────────────
PLOT_DPI        = 150
PLOT_STYLE      = "dark_background"
PALETTE         = {
    "primary":   "#6366f1",
    "secondary": "#a78bfa",
    "accent":    "#22d3ee",
    "success":   "#34d399",
    "warning":   "#f59e0b",
    "danger":    "#f43f5e",
    "bg":        "#0f172a",
    "surface":   "#1e293b",
    "text":      "#e2e8f0",
}

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST        = "0.0.0.0"
API_PORT        = 8000
