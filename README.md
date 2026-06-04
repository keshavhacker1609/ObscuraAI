# ObscuraAI 🔐

**Adversarial Demographic Leakage Auditing & Mitigation Framework**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

> Modern face recognition models silently encode sensitive demographic attributes — gender, age group, and ethnicity — inside their embedding vectors. **ObscuraAI** detects, quantifies, and suppresses this leakage through a fully automated three-module pipeline, backed by a live React dashboard, an async FastAPI orchestration layer, and a cryptographically tamper-evident audit trail.

---

## What It Does

Face recognition systems produce high-dimensional feature embeddings optimised for identity matching. A critical but often overlooked side-effect is **demographic attribute leakage** — adversarial classifiers can reliably extract sensitive traits from these embeddings with high accuracy, posing serious GDPR, AI Act, and biometric fairness compliance risks.

ObscuraAI provides end-to-end accountability for this vulnerability across three modules:

| Module | Purpose |
|--------|---------|
| **Module 1 — Adversarial Audit** | Trains LR + MLP attacker models against raw 512-dim embeddings to measure baseline demographic leakage via AUC-ROC, Privacy Leakage Score, and Mutual Information estimation |
| **Module 2 — Mitigation Engine** | Applies Gradient Reversal Layer (GRL) adversarial disentanglement with λ-sweep and Gaussian noise injection with σ-sweep — produces a full utility–privacy Pareto frontier |
| **Module 3 — Reporting** | Generates IEEE-standardised model cards with SHA-256 audit hashes, Max Demographic Disparity (MDD), Equalised Odds Difference (EOD), TAR@FAR utility metrics, and 7 publication-quality plots |

---

## Architecture

```
ObscuraAI/
├── ml_engine/                         # Core scientific pipeline
│   ├── config.py                      # Hyperparameters, paths, risk thresholds
│   ├── run_full_pipeline.py           # End-to-end CLI runner
│   ├── data/
│   │   └── generate_synthetic_celeba.py  # Synthetic 512-dim embedding generator
│   ├── module1_audit/
│   │   ├── attacker.py                # LR + MLP attackers, AUC / MI metrics
│   │   └── audit_pipeline.py         # Train/test split, orchestration, risk summary
│   ├── module2_mitigate/
│   │   ├── adversarial.py            # GRL network, AdversarialDisentangler, λ-sweep
│   │   ├── noise_injection.py        # Gaussian noise, SNR/drift metrics, σ-sweep
│   │   └── mitigation_pipeline.py
│   └── module3_report/
│       ├── metrics_engine.py         # Privacy, utility, fairness (MDD, EOD)
│       ├── visualizer.py             # 7 matplotlib publication-quality plots
│       └── report_generator.py      # IEEE model card + SHA-256 audit hash
├── backend/                           # FastAPI async orchestrator — port 8000
│   ├── main.py
│   ├── routes/
│   │   ├── pipeline.py               # POST /pipeline/run · GET /pipeline/status
│   │   ├── audit.py                  # /audit · /audit/summary · /roc-data · /per-class
│   │   ├── mitigate.py               # /mitigate · /comparison · sweep endpoints
│   │   ├── metrics.py                # /metrics/overview · /fairness · /utility
│   │   ├── report.py                 # /report · /report/download · /report/audit-trail
│   │   ├── visualizations.py         # /visualizations (structured array + PNG serving)
│   │   ├── logs.py                   # /logs · /logs/integrity — cryptographic chain
│   │   └── upload.py                 # POST /upload — custom .npz embedding upload
│   └── services/
│       ├── pipeline_runner.py        # Background thread, 12 audit checkpoints
│       └── audit_log.py             # SHA-256 blockchain-style tamper-evident log
└── frontend/                          # React 19 dashboard — port 3000
    └── src/
        ├── pages/
        │   ├── Dashboard.jsx         # KPI cards, live audit log, architecture flow
        │   ├── AuditPage.jsx         # ROC curves, per-class F1, radar chart
        │   ├── MitigationPage.jsx    # Before/after, Pareto sweep curves, TAR@FAR
        │   ├── ReportPage.jsx        # SHA-256 display, fairness chart, audit trail
        │   └── VisualizationsPage.jsx  # 7-plot gallery with category filter
        └── components/
            ├── Sidebar.jsx
            └── Badge.jsx
```

---

## Pipeline Flow

```
User → "Run Pipeline"
         │
         ▼
POST /pipeline/run
         │
         ▼  (background thread — polls GET /pipeline/status every 4 s)
┌────────────────────────────────────────────────────────────────────┐
│  Step 0  │  Load / generate synthetic 512-dim CelebA embeddings   │
│  Step 1  │  Module 1 — LR + MLP attacker training per attribute   │
│  Step 2  │  Module 2 — GRL adversarial (λ=0.8, 60 ep) + noise    │
│  Step 3a │  Fairness — per-group leakage, MDD, EOD                │
│  Step 3b │  Utility — TAR@FAR face verification simulation         │
│  Step 3c │  Visualize — 7 matplotlib publication-quality plots     │
│  Step 3d │  Report — IEEE model card + SHA-256 audit hash          │
└────────────────────────────────────────────────────────────────────┘
         │  Every step emits a SHA-256 chained audit log entry
         ▼
Dashboard unlocks: Audit · Mitigation · Report · Visualizations
```

---

## Key Features

### Module 1 — Adversarial Audit
- **Logistic Regression attacker** — linear lower-bound leakage estimate
- **MLP attacker** (256 → 128, LayerNorm + GELU + Dropout) — non-linear upper-bound
- **Privacy Leakage Score** — normalised above-chance metric `(AUC − chance) / (1 − chance)`
- **Mutual Information** estimation across top-20 embedding dimensions
- **Risk levels**: CRITICAL (>0.90) / HIGH (>0.75) / MEDIUM (>0.60) / LOW (>0.55) / NEGLIGIBLE
- **Per-class precision/recall/F1** from MLP classification report via `/audit/per-class`
- **Parametric ROC curves** (baseline vs. post-mitigation) for interactive Recharts rendering

### Module 2 — Mitigation Engine
- **Gradient Reversal Layer (GRL)** — `L_total = L_cosine + λ · L_adversarial`
  - Architecture: EmbeddingProjector (residual) → GRL → AttributeDiscriminator (per attribute)
  - λ-sweep over `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]`
- **Gaussian Noise Injection** — ε ~ N(0, σ²) on L2-normalised embeddings
  - σ-sweep over `[0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]`
  - Drift metrics: cosine similarity preservation, L2 distance, SNR (dB)
- **TAR@FAR utility** measurement at FAR ∈ {0.1 %, 1 %, 10 %}

### Module 3 — Reporting & Accountability
- **IEEE Model Card** (JSON) — `model_card_version`, `framework`, `dataset`, `privacy_analysis`, `mitigation`, `utility`, `fairness`, `recommendations`, `attribute_narratives`
- **SHA-256 Audit Hash** — deterministic hash over baseline summary + comparison table + fairness MDD + dataset metadata. Reproducible: same run always yields the same hash
- **Fairness Metrics**
  - *Max Demographic Disparity (MDD)* — max leakage gap across ethnicity subgroups
  - *Equalised Odds Difference (EOD)* — max balanced-accuracy gap across demographic groups
- **7 Publication-Quality Plots** — ROC curves, leakage bars, radar chart, trade-off curves, before/after comparison, adversarial training history, fairness disparity
- **Model Card JSON export** via `GET /report/download`

### Cryptographic Audit Trail
- **Blockchain-style chained log** — each entry stores `prev_hash` of the preceding entry's hash
- **Chain-integrity verification** via `GET /logs/integrity` — detects post-hoc tampering at the first broken sequence
- **12 checkpoints** emitted per pipeline run (dataset load → model card)
- **Live display** in the React dashboard with truncated chain hashes

---

## Quickstart

**Requirements:** Python 3.9+, Node.js 18+

### 1. Clone & set up Python environment
```bash
git clone https://github.com/keshavhacker1609/ObscuraAI.git
cd ObscuraAI

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Start the backend
```bash
uvicorn main:app --reload --port 8000 --app-dir backend
```
Interactive API docs → **http://localhost:8000/docs**

### 3. Start the frontend
```bash
cd frontend
npm install
npm start
```
Dashboard → **http://localhost:3000**

### 4. Run the pipeline
Open the dashboard and click **"Deep Run Pipeline"** (~3–5 min on CPU, full λ/σ sweep) or **"Fast Demo"** (~45 s, no sweep).

---

## CLI Usage

Run the full pipeline without the web interface:
```bash
python ml_engine/run_full_pipeline.py --dataset synthetic
```

Skip the λ/σ parameter sweep for a faster run:
```bash
python ml_engine/run_full_pipeline.py --dataset synthetic --skip-sweep
```

Upload and audit custom embeddings:
```bash
# 1. Upload a .npz file containing embeddings, gender, age_group, ethnicity arrays
curl -X POST http://localhost:8000/upload -F "file=@my_embeddings.npz"

# 2. Run pipeline on uploaded data
curl -X POST "http://localhost:8000/pipeline/run?dataset=uploaded&sweep=false"
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/pipeline/run` | Start the ML pipeline (`dataset`, `sweep` params) |
| `GET`  | `/pipeline/status` | Live status, progress %, run ID |
| `GET`  | `/audit` | Full baseline audit results |
| `GET`  | `/audit/summary` | Risk levels + leakage scores per attribute |
| `GET`  | `/audit/roc-data` | Parametric ROC curve points (baseline + post-mitigation) |
| `GET`  | `/audit/per-class` | Per-class precision / recall / F1 |
| `GET`  | `/mitigate` | Full mitigation results |
| `GET`  | `/mitigate/comparison` | Before/after AUC reduction table |
| `GET`  | `/mitigate/sweep/adversarial` | λ-sweep Pareto data |
| `GET`  | `/mitigate/sweep/noise` | σ-sweep Pareto data |
| `GET`  | `/metrics/overview` | Single-call dashboard overview |
| `GET`  | `/metrics/fairness` | Per-group leakage, MDD, EOD |
| `GET`  | `/metrics/utility` | TAR@FAR verification metrics |
| `GET`  | `/report` | Full IEEE model card JSON |
| `GET`  | `/report/download` | Download model card as `.json` |
| `GET`  | `/report/audit-trail` | Cryptographic audit log + chain integrity |
| `GET`  | `/visualizations` | All generated plots as structured array |
| `GET`  | `/visualizations/{name}` | Serve individual plot PNG |
| `GET`  | `/logs` | Audit log entries (`last_n` param) |
| `GET`  | `/logs/integrity` | Verify full SHA-256 chain |
| `POST` | `/upload` | Upload custom `.npz` embedding file |

---

## Metrics Reference

### Privacy
| Metric | Range | Interpretation |
|--------|-------|----------------|
| AUC-ROC | 0.5 – 1.0 | 0.5 = random baseline · 1.0 = perfect attribute extraction |
| Privacy Leakage Score | 0 – 1 | Normalised above-chance leakage |
| Mutual Information | ≥ 0 | Average MI across top-20 embedding dims |

### Utility
| Metric | Description |
|--------|-------------|
| TAR@FAR 0.1 % | True Accept Rate at 0.1 % False Accept Rate |
| TAR@FAR 1 % | True Accept Rate at 1 % False Accept Rate |
| Mean Genuine Similarity | Average cosine similarity for genuine pairs post-sanitisation |

### Fairness
| Metric | Description |
|--------|-------------|
| MDD | Max leakage gap across ethnicity subgroups |
| EOD | Max balanced-accuracy gap across demographic groups |

### Risk Levels
| Level | AUC Threshold | Action |
|-------|--------------|--------|
| CRITICAL | > 0.90 | Block deployment — apply GRL with λ ≥ 1.0 |
| HIGH | > 0.75 | Adversarial mitigation required before deployment |
| MEDIUM | > 0.60 | Noise injection (σ = 0.05) may suffice |
| LOW | > 0.55 | Monitor; re-audit after model updates |
| NEGLIGIBLE | ≤ 0.55 | Near-random — attribute information suppressed |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML / AI | PyTorch 2.0+, scikit-learn, NumPy, SciPy |
| Visualisation | Matplotlib (dark theme, 150 DPI) |
| Backend | FastAPI 0.111, Uvicorn 0.30, Pydantic v2 |
| Frontend | React 19, Recharts, Framer Motion, react-hot-toast, react-icons |
| Security | SHA-256 blockchain-style audit log, tamper-detection via chain integrity |
| Data | Synthetic 512-dim CelebA-inspired embeddings (N = 12,000, D = 512) |

---

## References

- Morales et al. (2020) *SensitiveNets: Learning Agnostic Representations with Application to Face Recognition.* **IEEE TPAMI**
- Terhorst et al. (2021) *Comprehensive Study of Face Recognition Biases Beyond Demographics.* **IEEE TIFS**
- Ganin & Lempitsky (2015) *Unsupervised Domain Adaptation by Backpropagation.* **ICML**
- Dhar et al. (2021) *PASS: Protected Attribute Suppression System for Mitigating Bias in Face Recognition.* **ICCV**
- Gong & Liu (2021) *Mitigating Face Recognition Bias via Group-Adaptive Classifier.* **CVPR**
- Mirjalili et al. (2020) *PrivacyNet: Semi-Adversarial Networks for Attribute Obfuscation.* **IEEE TIP**
- NIST (2023) *Face Recognition Vendor Test (FRVT) — Privacy Analysis Framework*
- Dwork et al. (2014) *The Algorithmic Foundations of Differential Privacy*

---

## Changelog

### v2.1.0
- Add `audit_log.py` — SHA-256 blockchain-style tamper-evident log with chain-integrity verification
- Add `logs.py` router — `GET /logs`, `GET /logs/integrity`
- Pipeline emits 12 real audit log entries with `run_id` tracking at every step
- Fix `/visualizations` — now returns structured array (was broken flat dict; frontend always showed empty)
- Fix `ReportPage` — correct API field mapping; live audit trail with chain hashes
- Add `GET /audit/roc-data` — parametric ROC curves for Recharts rendering
- Add `GET /audit/per-class` — per-class precision/recall/F1 from MLP report
- Add `GET /report/download` — downloadable model card JSON
- Add `GET /report/audit-trail` — live cryptographic audit trail endpoint
- Add Equalised Odds Difference (EOD) fairness metric
- SHA-256 `audit_hash` computed and embedded in every model card
- Replace dead legacy `leakage_detector.py` mock code with real ML engine delegation
- Fix redundant double train_test_split in `audit_pipeline.py`
- Rebrand frontend + sidebar to ObscuraAI v2.1

### v2.0.0
- Initial ObscuraAI release with 3-module pipeline, FastAPI backend, React dashboard

---

*Built by [keshavhacker1609](https://github.com/keshavhacker1609)*
