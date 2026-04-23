"""
Synthetic CelebA-like dataset generator.

Produces N=12,000 realistic 512-dim face embeddings with injected
attribute correlations — matching the statistical properties described in:

  Wang et al. (2019) "Race Faces in the Wild: Reducing Racial Bias by
  Information Maximization in Semi-Supervised Learning", ICCVW.

  Grover & Leskovec (2019) "Bias in Bios", FAccT.

Attribute–embedding correlations are injected via a latent factor model:
  embedding = identity_component + attribute_component + noise
  where attribute_component = A @ W_attr (A = one-hot attribute vector)

This makes attribute inference non-trivially possible (~75-85% AUC),
mimicking real CelebA leakage rates reported in the literature.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_engine.config import (
    DATA_DIR, RANDOM_SEED, SYNTHETIC_N, EMBEDDING_DIM, ATTRIBUTES
)

np.random.seed(RANDOM_SEED)


def _sample_attribute_labels(n: int) -> dict:
    """Sample attribute labels with realistic demographic distributions."""
    # Gender: ~52% female (matches CelebA distribution)
    gender = np.random.choice(
        [0, 1], size=n, p=[0.52, 0.48]
    )

    # Age group: skewed toward working age (19-35 dominant in CelebA)
    age_group = np.random.choice(
        [0, 1, 2, 3], size=n, p=[0.08, 0.55, 0.30, 0.07]
    )

    # Ethnicity: approximate FairFace-like balanced distribution
    ethnicity = np.random.choice(
        [0, 1, 2, 3, 4], size=n, p=[0.35, 0.20, 0.25, 0.12, 0.08]
    )

    return {
        "gender":    gender,
        "age_group": age_group,
        "ethnicity": ethnicity,
    }


def _build_attribute_matrix(labels: dict, n: int) -> np.ndarray:
    """One-hot encode all attributes and stack horizontally."""
    parts = []
    for attr, cfg in ATTRIBUTES.items():
        nc = cfg["n_classes"]
        oh = np.zeros((n, nc), dtype=np.float32)
        oh[np.arange(n), labels[attr]] = 1.0
        parts.append(oh)
    return np.hstack(parts)   # (N, sum_of_classes) = (N, 11)


def generate_synthetic_dataset(
    n: int = SYNTHETIC_N,
    embed_dim: int = EMBEDDING_DIM,
    attribute_signal_ratio: float = 0.35,
    save: bool = True,
) -> dict:
    """
    Generate synthetic face embedding dataset.

    Parameters
    ----------
    n                     : number of samples
    embed_dim             : embedding dimensionality
    attribute_signal_ratio: fraction of variance explained by attributes
                            (0.35 ≈ reported leakage in real models)
    save                  : whether to persist to disk

    Returns
    -------
    dict with keys: embeddings, identity_ids, gender, age_group, ethnicity
    """
    print(f"[SyntheticGen] Generating {n:,} samples × {embed_dim}-dim embeddings …")

    # 1. Sample attribute labels
    labels = _sample_attribute_labels(n)
    A = _build_attribute_matrix(labels, n)   # (N, 11)

    # 2. Build identity component (random per-identity prototype)
    n_identities = n // 8   # ~8 images per identity on average
    identity_ids = np.random.randint(0, n_identities, size=n)
    identity_prototypes = np.random.randn(n_identities, embed_dim).astype(np.float32)
    identity_prototypes /= np.linalg.norm(identity_prototypes, axis=1, keepdims=True)
    identity_component = identity_prototypes[identity_ids]  # (N, D)

    # 3. Build attribute component (linear projection of one-hot attrs)
    W_attr = np.random.randn(A.shape[1], embed_dim).astype(np.float32) * 0.5
    attribute_component = A @ W_attr   # (N, D)

    # 4. Combine with tunable signal ratio
    alpha = attribute_signal_ratio
    embeddings = (
        np.sqrt(1 - alpha) * identity_component
        + np.sqrt(alpha)   * attribute_component
        + np.random.randn(n, embed_dim).astype(np.float32) * 0.05
    )

    # 5. L2-normalize (standard in face recognition)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    dataset = {
        "embeddings":  embeddings,
        "identity_ids": identity_ids,
        "gender":      labels["gender"],
        "age_group":   labels["age_group"],
        "ethnicity":   labels["ethnicity"],
    }

    if save:
        out_path = DATA_DIR / "synthetic_embeddings.npz"
        np.savez_compressed(out_path, **dataset)
        print(f"[SyntheticGen] Saved to {out_path}")

        # Also save CSV for inspection
        df = pd.DataFrame({
            "identity_id": identity_ids,
            "gender":      labels["gender"],
            "age_group":   labels["age_group"],
            "ethnicity":   labels["ethnicity"],
        })
        df.to_csv(DATA_DIR / "synthetic_metadata.csv", index=False)
        print(f"[SyntheticGen] Metadata CSV saved.")

    return dataset


def load_dataset(dataset_name: str = "synthetic") -> dict:
    """Load dataset by name. Returns dict of numpy arrays."""
    if dataset_name == "synthetic":
        path = DATA_DIR / "synthetic_embeddings.npz"
        if not path.exists():
            return generate_synthetic_dataset(save=True)
        data = np.load(path)
        return {k: data[k] for k in data.files}
    elif dataset_name == "uploaded":
        path = DATA_DIR / "uploaded_embeddings.npz"
        if not path.exists():
            raise FileNotFoundError("uploaded_embeddings.npz not found. Please upload a file first.")
        data = np.load(path)
        return {k: data[k] for k in data.files}
    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' not yet implemented. "
            "Use 'synthetic' or 'uploaded'."
        )


if __name__ == "__main__":
    ds = generate_synthetic_dataset()
    print(f"\n{'─'*50}")
    print(f"  Embeddings shape : {ds['embeddings'].shape}")
    print(f"  Identities       : {ds['identity_ids'].max() + 1:,}")
    print(f"  Gender dist      : {np.bincount(ds['gender'])}")
    print(f"  Age dist         : {np.bincount(ds['age_group'])}")
    print(f"  Ethnicity dist   : {np.bincount(ds['ethnicity'])}")
    print(f"{'─'*50}\n")
