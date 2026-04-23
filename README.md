# AI Privacy Intelligence Framework 🛡️

A unified, full-stack framework for auditing, mitigating, and reporting attribute leakage in large-scale Face Recognition Models.

## Overview
This platform is designed to provide end-to-end accountability for privacy risks in embeddings. It systematically quantifies how much sensitive demographic information (like Gender, Age Group, Ethnicity) is leaked through feature vectors optimized for identity recognition. 

### Key Modules:
- **Module 1: Auditing** — Measures the exact privacy vulnerability using statistical modeling and adversary classification (LR and MLP).
- **Module 2: Mitigation** — Applies Adversarial Disentanglement (Gradient Reversal) and Noise Injection to mask sensitive traits without destroying identity recognition utility.
- **Module 3: Reporting** — Generates complete IEEE-standardized model cards, evaluates fairness disparity (MDD), and creates secure accountability logs.

## System Architecture

The ecosystem relies on an asynchronous backend architecture coupled with a clean React dashboard designed to Google Stitch visual specifications.

1. **`ml_engine/`:** Pure Python scientific core containing PyTorch architectures, synthetic dataset generation, plotting libraries, and mitigation utilities.  
2. **`backend/`:** A FastAPI application (`localhost:8000`) functioning as a background orchestrator. It manages asynchronous pipeline runs to allow users to start heavy ML jobs without blocking the HTTP thread, offering live polling endpoints.
3. **`frontend/`:** A modern Vite + React dashboard (`localhost:3000`) consuming the FastAPI data. It renders the data through beautiful Recharts and visually rich dashboards.

## Procedure: How the System Works

### 1. Initiation
The user interacts with the **Overview Dashboard** and clicks "Run Pipeline".
The React frontend sends a POST request to the backend `POST /pipeline/run?dataset=synthetic&sweep=true`.

### 2. Async Execution (Backend & ML Engine)
The backend accepts the request and spins up a background thread via `pipeline_runner.py`.
- **Dataset Creation:** The ML Engine creates an intricate 512-dim embedding set mimicking realistic face vectors (using an internal synthetic generative script).
- **Audit:** The system trains an attacker model against these vectors to learn demographics. It then tests them on a hold-out test set to get the "Baseline AUC".
- **Mitigation Sweeps:** The system runs an optimizer attempting two things natively via PyTorch: 
  - Erasing demographics through Adversarial Networks (GRL).
  - Injecting varying levels of Gaussian Noise.
- **Visuals & Metrics:** The system automatically drafts 7 comprehensive PNG charts (saved natively) and outputs standard IEEE fairness JSON metadata. 

### 3. Verification & Polling
Throughout this runtime, the React frontend continuously polls `GET /pipeline/status`.
The user watches the progress bar live on the dashboard. Upon completion, the UI automatically opens up navigation to the rest of the app.

### 4. Exploring the Artifacts
- **Auditing Page:** Retrieves the generated JSON from `/audit` and dynamically creates radar plots representing leakage.
- **Mitigation Page:** Compares before/after trade-off curves fetching from `/mitigate`.
- **Reporting Page:** Integrates the IEEE model card metrics, displaying cryptographic logs.
- **Visualizations Details:** Fetches raw generated PNGs from `/visualizations` and serves them with a custom zoom integration.

## Installation & Running Locally

Ensure you are located at the root of the project. A valid Python `venv` instance and Node installation are required.

### Terminal 1: Backend
```bash
cd backend
..\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Terminal 2: Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` to interact with the interface.

---
_Built with Vite, React, Recharts, PyTorch, and FastAPI._
