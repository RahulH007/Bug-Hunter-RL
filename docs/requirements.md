# Project Requirements — Autonomous Software Bug Hunter (RL)

---

## 1. Stakeholder Analysis

| ID | Stakeholder | Role | Primary Interest | Influence |
|---|---|---|---|---|
| SH-01 | ML Engineer (dev team) | Builder & maintainer | Model performance, reproducibility, clean API | High |
| SH-02 | DevOps / Platform Engineer | CI/CD & infra owner | Reliable pipelines, Docker images, rollback capability | High |
| SH-03 | QA / Test Lead | Consumer of bug-discovery results | High bug-discovery rate, low false-negative rate | Medium |
| SH-04 | Project Supervisor / Evaluator | Academic assessment | Documentation quality, SDG alignment, methodology rigour | High |
| SH-05 | Open-Source Community | Potential contributors | Reproducibility, clear contribution guide, clean code | Low–Medium |
| SH-06 | CI/CD System (GitHub Actions) | Automated executor | Deterministic builds, fast feedback, artefact upload | Medium |

---

## 2. Use Cases

### UC-01 — Train the RL Agent
**Actor:** ML Engineer (SH-01)  
**Pre-condition:** `requirements.txt` installed; graph environment defined.  
**Main flow:**
1. Engineer invokes `python train.py --episodes N --seed S`.
2. System builds the graph environment and Q-Learning agent.
3. Agent explores the graph for N episodes, updating Q-values.
4. System logs run record (UUID, hyperparameters, metrics) to JSON/CSV and MLflow.
5. System serialises the trained policy to `models/policy_v1.pkl`.  
**Post-condition:** Trained policy available; MLflow experiment updated.  
**Alternate flow:** `--mlflow-uri` overrides default local tracking server.

---

### UC-02 — Evaluate RL Agent vs Baseline
**Actor:** ML Engineer / QA Lead (SH-01, SH-03)  
**Pre-condition:** Trained policy exists at `models/policy_v1.pkl`.  
**Main flow:**
1. Engineer invokes `python evaluate.py --episodes M`.
2. System runs both agents on identical episode seeds.
3. System prints side-by-side comparison table (reward, steps, discovery rate).
4. System saves two plots and `logs/eval_metrics.json`.
5. System logs evaluation results to MLflow under the `bug-hunter-rl-eval` experiment.  
**Post-condition:** Evaluation artefacts committed to `plots/`; metrics in MLflow.

---

### UC-03 — Visualise Agent in Real Time
**Actor:** QA Lead / Project Supervisor (SH-03, SH-04)  
**Pre-condition:** Streamlit and a trained policy are available.  
**Main flow:**
1. User runs `streamlit run app.py` (or `docker compose up`).
2. User selects agent type, episode seed, bug-spawn multiplier in sidebar.
3. User clicks "Run Simulation".
4. Dashboard renders the graph live, colouring nodes by agent position and bug status.  
**Alternate flow:** User can switch to Random-Walker baseline at any time.

---

### UC-04 — Monitor System Health
**Actor:** DevOps Engineer / CI pipeline (SH-02, SH-06)  
**Pre-condition:** At least two training runs logged in `logs/results.json`.  
**Main flow:**
1. Engineer (or CI) runs `python monitoring.py`.
2. System checks discovery-rate drift, waiting-time creep, reward regression, training-time spike.
3. System prints colour-coded health report.
4. System exits 0 (GREEN), 1 (WARN), or 2 (CRITICAL).

---

### UC-05 — Roll Back a Policy
**Actor:** ML Engineer / DevOps Engineer (SH-01, SH-02)  
**Pre-condition:** At least one snapshot exists in `models/snapshots/`.  
**Main flow:**
1. Engineer runs `python rollback.py list` to see available snapshots.
2. Engineer runs `python rollback.py restore <snapshot_id>`.
3. System auto-saves the current policy as a pre-restore snapshot.
4. System copies the selected snapshot to `models/policy_v1.pkl`.

---

### UC-06 — Run Full Pipeline via DVC
**Actor:** ML Engineer / CI system (SH-01, SH-06)  
**Pre-condition:** DVC initialised; `params.yaml` configured.  
**Main flow:**
1. Engineer edits `params.yaml`.
2. Engineer runs `dvc repro`.
3. DVC detects changed stages and re-runs only what's necessary (train, evaluate).
4. DVC records new `dvc.lock` and updated metrics for diff comparison.

---

### UC-07 — Containerised Deployment
**Actor:** DevOps Engineer (SH-02)  
**Pre-condition:** Docker and Docker Compose installed.  
**Main flow:**
1. Engineer runs `docker compose up`.
2. MLflow tracking server starts on port 5000.
3. Streamlit dashboard starts on port 8501, pointing at the MLflow server.
4. Engineer can run `docker compose --profile training up trainer` for a container-isolated training job.

---

## 3. Functional Requirements

| ID | Requirement | Priority | Source |
|---|---|---|---|
| FR-01 | The system shall train a tabular Q-Learning agent on a configurable graph environment. | Must | UC-01 |
| FR-02 | The system shall accept CLI flags for all hyperparameters (episodes, lr, gamma, epsilon, seed). | Must | UC-01 |
| FR-03 | The system shall log every training run with a UUID, UTC timestamp, hyperparameters, and metrics to JSON and CSV. | Must | UC-01 |
| FR-04 | The system shall log training runs and evaluation results to an MLflow tracking server. | Must | UC-01, UC-02 |
| FR-05 | The system shall evaluate the RL agent against a Random-Walker baseline on identical episode seeds. | Must | UC-02 |
| FR-06 | The system shall produce at least two evaluation plots (reward curve, bugs-found curve). | Must | UC-02 |
| FR-07 | The system shall export evaluation metrics to `logs/eval_metrics.json` for DVC metric tracking. | Must | UC-06 |
| FR-08 | The system shall provide a Streamlit dashboard with live graph rendering and a step-by-step action log. | Must | UC-03 |
| FR-09 | The system shall provide a monitoring script that checks for policy drift, reward regression, and training-time spikes. | Must | UC-04 |
| FR-10 | The system shall provide a rollback utility to save, list, restore, and delete policy snapshots. | Must | UC-05 |
| FR-11 | The system shall define a DVC pipeline (`dvc.yaml`) linking train and evaluate stages with parameter files. | Must | UC-06 |
| FR-12 | The system shall be fully containerisable via a `Dockerfile` and `docker-compose.yml`. | Must | UC-07 |
| FR-13 | The system shall support an MLflow tracking URI environment variable (`MLFLOW_TRACKING_URI`). | Should | UC-07 |
| FR-14 | The system shall expose per-episode window metrics as a time series in MLflow (step-based logging). | Should | UC-01 |
| FR-15 | The system shall deterministically reproduce results given the same seed across all components. | Must | UC-01, UC-02 |

---

## 4. Non-Functional Requirements

| ID | Category | Requirement | Metric / Acceptance Criteria |
|---|---|---|---|
| NFR-01 | Performance | Training 2000 episodes on the 18-node graph shall complete in under 60 seconds on standard laptop hardware. | ≤ 60 s (measured via `train_seconds` in log) |
| NFR-02 | Accuracy | The RL agent shall achieve ≥ 90% bug discovery rate after 5000 training episodes. | `bug_discovery_rate ≥ 0.90` in `eval_metrics.json` |
| NFR-03 | Reproducibility | Identical seeds shall produce byte-for-byte identical Q-tables and episode rewards. | Deterministic test in CI |
| NFR-04 | Maintainability | All source files shall pass flake8 with `--max-line-length=120`. | CI lint step green |
| NFR-05 | Portability | The system shall run on Python 3.10+ on Linux, macOS, and Windows. | CI matrix covers `ubuntu-latest` |
| NFR-06 | Scalability | The graph size shall be configurable from 5 to 18 nodes without code changes. | `--n-nodes` flag tested in smoke train |
| NFR-07 | Observability | Every training run shall produce a structured log entry readable by the monitoring script. | `monitoring.py` exits 0 after any valid run |
| NFR-08 | Recoverability | A failed or regressed policy can be rolled back in ≤ 2 CLI commands. | `rollback.py restore <id>` tested manually |
| NFR-09 | Security | No secrets, credentials, or API keys shall appear in source code or committed files. | Pre-commit / CI secret scan |
| NFR-10 | Documentation | A `docs/requirements.md` (this file), `CONTRIBUTING.md`, and complete README shall be present. | File existence checked in CI |
| NFR-11 | Container startup | `docker compose up` shall bring the Streamlit dashboard to a healthy state in ≤ 60 seconds. | Docker healthcheck passes |

---

## 5. Feasibility Study

### 5.1 Technical Feasibility
**Status: Feasible.**  
The problem space (18 nodes, ~288 discrete states) is well within the capacity of tabular Q-Learning. All required libraries (NumPy, NetworkX, Streamlit, MLflow, DVC) are mature, open-source, and pip-installable. Docker containerisation is straightforward given the lightweight Python stack.

### 5.2 Economic Feasibility
**Status: Fully viable at zero cost.**  
All tools are open-source. GitHub Actions provides free CI minutes for public repositories. MLflow and DVC use local file storage by default (no cloud spend required). Optional S3/GCS DVC remotes incur minimal costs only at scale.

### 5.3 Schedule Feasibility
**Status: Achieved.**  
The core RL system (env, agent, training, evaluation, dashboard) was built in the initial sprint. MLOps infrastructure (MLflow, DVC, CI/CD, monitoring, rollback) was layered on in a second sprint. Total development time is consistent with a 4–6 week academic project.

### 5.4 Operational Feasibility
**Status: Feasible.**  
The monitoring script and rollback utility lower the operational overhead to a single CLI call each. Docker Compose enables one-command deployment without manual environment management.

---

## 6. Constraints

| ID | Constraint | Type | Impact |
|---|---|---|---|
| C-01 | Graph is limited to 18 nodes (MODULE_CATALOGUE size). | Technical | Limits environment realism; acceptable for academic scope. |
| C-02 | Tabular Q-Learning requires a discrete, compact state space. | Technical | Cannot scale to 100+ nodes without switching to DQN. |
| C-03 | No external data sources; all bug placements are synthetic. | Scope | Results are not directly comparable to real-world defect data. |
| C-04 | MLflow persists run data locally in `./mlruns`; no remote configured by default. | Infrastructure | Team members must share `mlruns/` manually or configure a remote server. |
| C-05 | DVC remote storage is not pre-configured; models are not automatically synced. | Infrastructure | Users must run `dvc remote add` before `dvc push`. |
| C-06 | Streamlit reruns the full Python script on every interaction (no persistent server). | Framework | Limits the complexity of real-time state that can be maintained in the UI. |

---

## 7. Trade-offs

| Decision | Option Chosen | Alternative(s) Considered | Reason |
|---|---|---|---|
| RL algorithm | Tabular Q-Learning | DQN, PPO, A2C | ~288 states fit a Q-table; fully explainable, fast, no GPU needed. |
| State encoding | 4-tuple (node, time, tested, tested-flag) | Raw node + full history | Keeps Q-table compact while preserving the oscillation-prevention flag. |
| MLOps platform | MLflow (local file store) | Weights & Biases, Neptune | Zero infrastructure dependency; self-hostable; free. |
| Pipeline versioning | DVC | MLflow Projects, Kubeflow | DVC is language-agnostic, Git-native, and lightweight. |
| Containerisation | Docker Compose (multi-service) | Single Dockerfile | Separates app and MLflow server cleanly for local and CI use. |
| Bug logging | Custom JSON + CSV + MLflow | MLflow only | Custom logs survive without an MLflow server; provide plain-text auditability. |

---

## 8. Risk Analysis

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-01 | Q-table fails to converge (policy stuck in suboptimal behaviour). | Medium | High | Monitor `bug_discovery_rate` trend; increase episodes; tune epsilon-decay. Rollback utility allows reverting to a known-good policy. |
| R-02 | MLflow tracking URI unreachable in CI, causing CI failure. | Low | Medium | MLflow defaults to local file store — no server needed in CI. |
| R-03 | DVC pipeline becomes stale (params changed without `dvc repro`). | Medium | Medium | `dvc status` in CI; PR checklist includes DVC verification. |
| R-04 | Docker image bloat due to large pip packages (e.g., mlflow). | Low | Low | Multi-stage build or `.dockerignore` can be added if image size becomes an issue. |
| R-05 | Numpy / NetworkX API breaking changes across major versions. | Low | Medium | Pinned version ranges in `requirements.txt`; dependabot can alert on updates. |
| R-06 | Streamlit rerun loop causes high CPU on slow machines. | Medium | Low | `step_delay` slider lets users reduce auto-step frequency. |
| R-07 | Policy file corruption on crash during training. | Low | High | `rollback.py save` before any experiment; training writes atomically to a path. |

---

## 9. Traceability Matrix

Maps Use Cases → Functional Requirements → Source Files.

| Use Case | Functional Requirements | Primary Files |
|---|---|---|
| UC-01 (Train) | FR-01, FR-02, FR-03, FR-04, FR-14, FR-15 | `train.py`, `agent.py`, `env.py` |
| UC-02 (Evaluate) | FR-05, FR-06, FR-07, FR-04 | `evaluate.py`, `agent.py`, `env.py` |
| UC-03 (Visualise) | FR-08 | `app.py`, `env.py`, `agent.py` |
| UC-04 (Monitor) | FR-09, FR-03 | `monitoring.py`, `logs/results.json` |
| UC-05 (Rollback) | FR-10 | `rollback.py`, `models/snapshots/` |
| UC-06 (DVC Pipeline) | FR-11, FR-07 | `dvc.yaml`, `params.yaml` |
| UC-07 (Docker) | FR-12, FR-13 | `Dockerfile`, `docker-compose.yml` |

| Non-Functional Req | Verified By |
|---|---|
| NFR-01 (Performance) | `train_seconds` in `logs/results.json` |
| NFR-02 (Accuracy) | `logs/eval_metrics.json` → `rl_discovery_rate` |
| NFR-03 (Reproducibility) | CI smoke test (same seed → same Q-table) |
| NFR-04 (Maintainability) | CI lint job (flake8) |
| NFR-05 (Portability) | CI runs on `ubuntu-latest` |
| NFR-07 (Observability) | `monitoring.py` exit 0 in CI |
| NFR-08 (Recoverability) | `rollback.py restore` smoke test |
| NFR-10 (Documentation) | File existence checked in CI |
