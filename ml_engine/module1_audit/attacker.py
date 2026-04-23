"""
Attribute Leakage Auditing Module — Attacker Models

Implements two adversarial attacker classifiers that attempt to infer
sensitive attributes from face embeddings. This follows the threat model
described in:

  Morales et al. (2020) "SensitiveNets: Learning Agnostic Representations
  with Application to Face Recognition." IEEE TPAMI.

  Terhorst et al. (2021) "Comprehensive Study of Face Recognition Biases
  Beyond Demographics." IEEE TIFS.

Attackers:
  1. Logistic Regression (linear, serves as lower-bound leakage estimate)
  2. Multi-Layer Perceptron (non-linear, upper-bound leakage estimate)

Metrics:
  - Accuracy, Balanced Accuracy
  - AUC-ROC (macro OvR for multi-class)
  - Mutual Information (MI) estimation
  - Privacy Leakage Score = normalized AUC above random baseline
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, classification_report
)
from sklearn.feature_selection import mutual_info_classif

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import (
    LR_MAX_ITER, MLP_HIDDEN, MLP_EPOCHS, MLP_LR, MLP_BATCH_SIZE,
    RANDOM_SEED, ATTRIBUTES, RESULTS_DIR
)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ─── MLP Attacker ─────────────────────────────────────────────────────────────

class MLPAttacker(nn.Module):
    """
    2-layer MLP attribute inference attacker.
    Input: face embedding (512-dim)
    Output: attribute class logits
    """
    def __init__(self, input_dim: int, hidden: tuple, n_classes: int):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    embed_dim: int,
) -> MLPAttacker:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPAttacker(embed_dim, MLP_HIDDEN, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MLP_EPOCHS
    )
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=MLP_BATCH_SIZE, shuffle=True
    )

    model.train()
    for epoch in range(MLP_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{MLP_EPOCHS} | loss={epoch_loss/len(loader):.4f}")

    return model


def _predict_mlp(model: MLPAttacker, X: np.ndarray) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
    return preds, probs


# ─── Leakage Metrics ──────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y_true)
            return float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        return 0.5


def compute_privacy_leakage_score(auc: float, n_classes: int) -> float:
    """
    Normalized leakage score ∈ [0, 1].
    0 = no leakage (random baseline), 1 = perfect attribute inference.
    Formula: (AUC - chance) / (1 - chance), where chance = 1/n_classes for uniform.
    """
    chance = 1.0 / n_classes
    return max(0.0, (auc - chance) / (1.0 - chance + 1e-8))


def compute_mutual_information(X: np.ndarray, y: np.ndarray) -> float:
    """Average MI across top-20 embedding dimensions."""
    # Sample 20 dims for speed (full 512 is too slow for MI)
    idx = np.random.choice(X.shape[1], size=min(20, X.shape[1]), replace=False)
    mi = mutual_info_classif(X[:, idx], y, random_state=RANDOM_SEED)
    return float(mi.mean())


# ─── Main Audit Function ───────────────────────────────────────────────────────

def run_attribute_audit(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train_dict: dict,   # {attr_name: train_labels}
    y_test_dict:  dict,   # {attr_name: test_labels}
    embed_dim: int = 512,
    tag: str = "baseline",
) -> dict:
    """
    Train LR + MLP attackers for each attribute and compute leakage metrics.

    Returns
    -------
    results dict with per-attribute metrics for both attacker types.
    """
    results = {}
    embed_dim = X_train.shape[1]

    for attr, cfg in ATTRIBUTES.items():
        n_classes = cfg["n_classes"]
        y_tr = y_train_dict[attr]
        y_te = y_test_dict[attr]

        print(f"\n  [Audit] Attribute: {attr.upper()} ({n_classes} classes)")

        # ── Logistic Regression Attacker ──────────────────────────────────────
        print("    Training Logistic Regression attacker …")
        lr_attacker = LogisticRegression(
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_SEED,
            C=1.0,
            solver="lbfgs",
        )
        lr_attacker.fit(X_train, y_tr)
        lr_preds = lr_attacker.predict(X_test)
        lr_probs = lr_attacker.predict_proba(X_test)
        lr_acc  = float(accuracy_score(y_te, lr_preds))
        lr_bacc = float(balanced_accuracy_score(y_te, lr_preds))
        lr_auc  = compute_auc(y_te, lr_probs, n_classes)
        lr_mi   = compute_mutual_information(X_test, y_te)
        lr_leak = compute_privacy_leakage_score(lr_auc, n_classes)

        print(f"    LR  → Acc={lr_acc:.3f} | BalAcc={lr_bacc:.3f} | AUC={lr_auc:.3f} | Leakage={lr_leak:.3f}")

        # ── MLP Attacker ───────────────────────────────────────────────────────
        print("    Training MLP attacker …")
        mlp_model = _train_mlp(X_train, y_tr, n_classes, embed_dim)
        mlp_preds, mlp_probs = _predict_mlp(mlp_model, X_test)
        mlp_acc  = float(accuracy_score(y_te, mlp_preds))
        mlp_bacc = float(balanced_accuracy_score(y_te, mlp_preds))
        mlp_auc  = compute_auc(y_te, mlp_probs, n_classes)
        mlp_leak = compute_privacy_leakage_score(mlp_auc, n_classes)

        print(f"    MLP → Acc={mlp_acc:.3f} | BalAcc={mlp_bacc:.3f} | AUC={mlp_auc:.3f} | Leakage={mlp_leak:.3f}")

        # ── Per-class report ──────────────────────────────────────────────────
        report = classification_report(y_te, mlp_preds, output_dict=True)

        results[attr] = {
            "n_classes": n_classes,
            "labels":    cfg["labels"],
            "logistic_regression": {
                "accuracy":         lr_acc,
                "balanced_accuracy": lr_bacc,
                "auc_roc":          lr_auc,
                "leakage_score":    lr_leak,
                "mutual_information": lr_mi,
            },
            "mlp": {
                "accuracy":         mlp_acc,
                "balanced_accuracy": mlp_bacc,
                "auc_roc":          mlp_auc,
                "leakage_score":    mlp_leak,
                "mutual_information": lr_mi,   # same MI measure
            },
            "best_attacker_auc":     max(lr_auc, mlp_auc),
            "best_leakage_score":    max(lr_leak, mlp_leak),
            "classification_report": report,
        }

        # Risk level
        best_auc = results[attr]["best_attacker_auc"]
        from ml_engine.config import RISK_THRESHOLDS
        for level, threshold in RISK_THRESHOLDS.items():
            if best_auc >= threshold:
                results[attr]["risk_level"] = level
                break
        else:
            results[attr]["risk_level"] = "NEGLIGIBLE"

    # Aggregate leakage index
    all_leak = [r["best_leakage_score"] for r in results.values()]
    results["__aggregate__"] = {
        "mean_leakage_score": float(np.mean(all_leak)),
        "max_leakage_score":  float(np.max(all_leak)),
        "tag":                tag,
    }

    return results
