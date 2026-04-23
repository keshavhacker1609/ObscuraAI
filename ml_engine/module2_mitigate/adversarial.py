"""
Module 2: Adversarial Attribute Disentanglement

Implements gradient-reversal-based adversarial training to remove sensitive
attribute information from face embeddings while preserving identity utility.

Architecture based on:
  Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by Backpropagation", ICML.
  Wadsworth et al. (2018) "Achieving Fairness through Adversarial Learning", FairML.
  Morales et al. (2020) "SensitiveNets", IEEE TPAMI.

Network:
  EmbeddingEncoder → ProjectionHead (identity-preserving)
                  ↘ GradientReversal → AttributeDiscriminator (per attribute)

Loss:
  L_total = L_cosine_identity + λ * L_adversarial_attributes
  (λ controls privacy–utility trade-off)
"""

import numpy as np
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import (
    ADV_EPOCHS, ADV_LR, ADV_LAMBDA, ADV_LAMBDA_GRID,
    EMBEDDING_DIM, ATTRIBUTES, RANDOM_SEED, RESULTS_DIR, MODELS_DIR
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── Gradient Reversal Layer ───────────────────────────────────────────────────

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ─── Network Components ────────────────────────────────────────────────────────

class EmbeddingProjector(nn.Module):
    """Maps original embedding to privacy-sanitised space (same dim)."""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        out = self.net(x) + x   # residual
        return nn.functional.normalize(out, p=2, dim=1)


class AttributeDiscriminator(nn.Module):
    """Per-attribute discriminator head."""
    def __init__(self, dim: int, n_classes: int, alpha: float):
        super().__init__()
        self.reversal = GradientReversal(alpha)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x_rev = self.reversal(x)
        return self.classifier(x_rev)


class AdversarialDisentangler(nn.Module):
    """Full adversarial disentanglement network."""
    def __init__(self, embed_dim: int, lam: float):
        super().__init__()
        self.projector = EmbeddingProjector(embed_dim)
        self.discriminators = nn.ModuleDict({
            attr: AttributeDiscriminator(embed_dim, cfg["n_classes"], alpha=lam)
            for attr, cfg in ATTRIBUTES.items()
        })

    def forward(self, x):
        z = self.projector(x)
        attr_logits = {
            attr: disc(z)
            for attr, disc in self.discriminators.items()
        }
        return z, attr_logits


# ─── Training ──────────────────────────────────────────────────────────────────

def train_adversarial_disentangler(
    X_train: np.ndarray,
    y_train_dict: dict,
    lam: float = ADV_LAMBDA,
    epochs: int = ADV_EPOCHS,
    verbose: bool = True,
) -> tuple:
    """
    Train the adversarial disentangler.

    Returns
    -------
    (model, history_dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [Adversarial] λ={lam:.2f} | device={device} | epochs={epochs}")

    embed_dim = X_train.shape[1]
    model = AdversarialDisentangler(embed_dim, lam).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ADV_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Build tensors
    X_t = torch.tensor(X_train, dtype=torch.float32)
    label_tensors = {
        attr: torch.tensor(y_train_dict[attr], dtype=torch.long)
        for attr in ATTRIBUTES
    }
    dataset = TensorDataset(X_t, *[label_tensors[a] for a in ATTRIBUTES])
    loader  = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    ce = nn.CrossEntropyLoss()
    attr_names = list(ATTRIBUTES.keys())

    history = {"identity_loss": [], "attr_loss": [], "total_loss": []}

    for epoch in range(epochs):
        model.train()
        ep_id_loss, ep_attr_loss = 0.0, 0.0

        for batch in loader:
            xb = batch[0].to(device)
            ybs = [batch[i+1].to(device) for i in range(len(attr_names))]

            optimizer.zero_grad()
            z, attr_logits = model(xb)

            # Identity loss: cosine similarity to original embedding (self-supervised)
            id_loss = 1.0 - (xb * z).sum(dim=1).mean()

            # Attribute adversarial loss
            attr_loss = sum(
                ce(attr_logits[attr], ybs[i])
                for i, attr in enumerate(attr_names)
            ) / len(attr_names)

            total_loss = id_loss + lam * attr_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_id_loss   += id_loss.item()
            ep_attr_loss += attr_loss.item()

        scheduler.step()
        history["identity_loss"].append(ep_id_loss / len(loader))
        history["attr_loss"].append(ep_attr_loss / len(loader))
        history["total_loss"].append(
            (ep_id_loss + lam * ep_attr_loss) / len(loader)
        )

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch+1:3d}/{epochs} | "
                f"ID={ep_id_loss/len(loader):.4f} | "
                f"Attr={ep_attr_loss/len(loader):.4f}"
            )

    return model, history


def apply_disentangler(
    model: AdversarialDisentangler,
    X: np.ndarray,
) -> np.ndarray:
    """Apply trained disentangler to produce sanitised embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        z, _ = model(X_t)
        return z.cpu().numpy()


def run_lambda_sweep(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train_dict: dict,
    y_test_dict:  dict,
    audit_fn,
    lambda_grid: list = ADV_LAMBDA_GRID,
) -> list:
    """
    Sweep λ values and compute utility-privacy trade-off for each point.
    Returns list of dicts: {lambda, mean_leakage_score, mean_attacker_auc}
    """
    print(f"\n[Adversarial] Running λ-sweep over {lambda_grid} …")
    sweep_results = []

    for lam in lambda_grid:
        print(f"\n  ── λ = {lam:.2f} ──────────────────────────")
        model, _ = train_adversarial_disentangler(
            X_train, y_train_dict, lam=lam, epochs=max(10, ADV_EPOCHS // 3),
            verbose=False
        )
        X_train_san = apply_disentangler(model, X_train)
        X_test_san  = apply_disentangler(model, X_test)

        audit = audit_fn(X_train_san, X_test_san, y_train_dict, y_test_dict, tag=f"adv_lam{lam}")
        agg = audit["__aggregate__"]

        # Compute identity utility: cosine similarity to original embeddings
        cos_sim = float(np.mean(np.sum(X_test * X_test_san, axis=1)))

        sweep_results.append({
            "lambda":             lam,
            "mean_leakage_score": agg["mean_leakage_score"],
            "max_leakage_score":  agg["max_leakage_score"],
            "identity_utility":   cos_sim,
        })
        print(f"    → Leakage={agg['mean_leakage_score']:.3f}  Utility={cos_sim:.3f}")

    return sweep_results
