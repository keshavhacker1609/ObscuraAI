# ObscuraAI 🕶️

### Adversarial Disentanglement Framework for Demographic Leakage Auditing and Mitigation in Face Recognition Systems

> Modern face recognition models silently encode sensitive demographic attributes — gender, age, ethnicity — inside their embedding vectors. **ObscuraAI** detects, quantifies, and eliminates this leakage through a fully automated three-stage pipeline, backed by a live React dashboard and an async FastAPI orchestration layer.

---

## 🔍 What It Does

Face recognition systems produce high-dimensional feature embeddings optimized for identity matching. A critical but overlooked side-effect is **attribute leakage** — adversarial classifiers can reliably extract demographic traits from these embeddings with high accuracy, posing serious GDPR and AI fairness compliance risks.

**ObscuraAI** provides end-to-end accountability for this privacy vulnerability:

| Module | Purpose |
|--------|---------|
| **Module 1 — Audit** | Trains LR + MLP attacker models against raw embeddings to measure baseline demographic leakage (AUC scores per attribute) |
| **Module 2 — Mitigate** | Applies Adversarial Disentanglement (Gradient Reversal Layer) and Gaussian Noise Injection to sanitize embeddings |
| **Module 3 — Report** | Generates IEEE-standardized model cards, Max Demographic Disparity (MDD) scores, TAR@FAR utility metrics, and 7 publication-quality plots |

---

## 🏗️ System Architecture

```
ObscuraAI/
├── ml_engine/              # Core scientific pipeline (PyTorch, NumPy, scikit-learn)
│   ├── module1_audit/      # Attacker models, leakage AUC computation
│   ├── module2_mitigate/   # Adversarial GRL, noise injection, lambda sweeps
│   ├── module3_report/     # Fairness engine, visualizer, model card generator
│   └── data/               # Synthetic 512-dim face embedding generator
├── backend/                # FastAPI async orchestrator (port 8000)
│   ├── routes/             # /audit, /mitigate, /pipeline, /report, /visualizations
│   └── services/           # Background pipeline runner, leakage detector
└── frontend/               # Vite + React dashboard (port 3000)
    └── src/
        ├── pages/          # Dashboard, Audit, Mitigation, Report, Visualizations
        └── components/     # LeakageRadar, TradeoffChart, MetricCard, RiskBadge
```

---

## ⚙️ How the Pipeline Works

1. **User clicks "Run Pipeline"** on the React dashboard
2. Frontend sends `POST /pipeline/run?dataset=synthetic&sweep=true`
3. Backend spawns a background thread and runs all 3 modules asynchronously
4. Frontend continuously polls `GET /pipeline/status` — live progress bar updates in real-time
5. On completion, the full dashboard unlocks:
   - **Audit Page** — demographic leakage radar charts per attribute
   - **Mitigation Page** — before/after utility vs. privacy trade-off curves
   - **Report Page** — IEEE model card with fairness scores and cryptographic logs
   - **Visualizations** — 7 auto-generated PNG plots with zoom integration

---

## 🚀 Running Locally

Requires Python 3.9+ with `venv` and Node.js.

### Terminal 1 — Backend
```bash
cd backend
..\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Terminal 2 — Frontend
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** to interact with the dashboard.

---

## 🧪 Run Pipeline Directly (CLI)
```bash
python ml_engine/run_full_pipeline.py --dataset synthetic
```

Add `--skip-sweep` for a faster run without the full λ/σ parameter sweep.

---

## 📊 Key Metrics Produced

- **Baseline Leakage AUC** — per demographic attribute (gender, age, ethnicity)
- **Post-Mitigation AUC** — after adversarial disentanglement and noise injection
- **Max Demographic Disparity (MDD)** — IEEE fairness compliance score
- **TAR@FAR** — identity verification utility preserved after sanitization
- **Risk Level** — overall privacy risk classification (Low / Medium / High / Critical)

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML Core | PyTorch, scikit-learn, NumPy, Matplotlib |
| Backend | FastAPI, Uvicorn, Python 3.9+ |
| Frontend | React, Vite, Recharts, CSS3 |
| Data | Synthetic 512-dim CelebA-inspired embeddings |

---

_Built with PyTorch · FastAPI · React · Recharts · Vite_
