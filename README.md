# Autonomous Software Bug Hunter — Reinforcement Learning

> An RL agent that learns to navigate a software-module graph and discover bugs efficiently — with MLflow experiment tracking, DVC pipeline versioning, CI/CD automation, Docker deployment, and an interactive Streamlit demo.

This project models a microservice-style codebase as a connected graph of modules (Auth, Database, API Gateway, Payment, …) and trains a tabular **Q-Learning** agent to traverse it, test nodes for bugs, and maximise reward while minimising steps. The trained agent is benchmarked against a Random-Walker baseline and visualised live in a web dashboard.

---

## Project Structure

```
bug_hunter_rl/
├── env.py                  # Custom graph environment (no Gym dependency)
├── agent.py                # Q-Learning agent + Random-Walker baseline
├── train.py                # Training loop + MLflow + MLOps logging
├── evaluate.py             # RL vs baseline comparison + MLflow + plots
├── monitoring.py           # Operational health checks (drift, regression, spikes)
├── rollback.py             # Policy snapshot manager (save / restore / list)
├── app.py                  # Streamlit interactive dashboard
│
├── params.yaml             # DVC parameter file — single source of truth for config
├── dvc.yaml                # DVC pipeline (train → evaluate stages)
├── Dockerfile              # Container image for the Streamlit app
├── docker-compose.yml      # Multi-service stack (app + MLflow server)
│
├── requirements.txt        # pip dependencies
├── environment.yml         # conda environment (alternative)
├── .dvcignore
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml          # CI: lint → smoke-train → evaluate → DVC pipeline
│   │   └── cd.yml          # CD: Docker build & push to GHCR on main / tags
│   ├── pull_request_template.md
│   └── CODEOWNERS
│
├── CONTRIBUTING.md         # Branching strategy, commit conventions, PR workflow
├── docs/
│   └── requirements.md     # Stakeholders, use cases, FR/NFR, risks, traceability
│
├── models/                 # Saved policies (created by train.py)
│   └── snapshots/          # Versioned policy snapshots (managed by rollback.py)
├── logs/                   # MLOps logs (results.json, results.csv, eval_metrics.json)
├── plots/                  # Evaluation plots (created by evaluate.py)
└── mlruns/                 # MLflow local run store (created at runtime)
```

---

## Project Components

| Component | File | Description |
|---|---|---|
| **Simulator** | `env.py` | Connected graph of 18 modules; bugs spawned with per-module risk weights. Discrete state, neighbour-indexed actions, reward shaping. |
| **RL Agent** | `agent.py` | Tabular Q-Learning with epsilon-greedy + exponential decay. Saves/loads policy as a pickle. |
| **Baseline Agent** | `agent.py` | Random Walker — picks a uniformly random neighbour every step. |
| **Training + MLOps** | `train.py` | Trains the agent; logs to MLflow (params, metrics, step-series, artifact) and to `logs/results.json` / `logs/results.csv`. |
| **Evaluation** | `evaluate.py` | Runs both agents on identical seeds, prints comparison table, saves plots, writes `logs/eval_metrics.json`, logs to MLflow. |
| **Monitoring** | `monitoring.py` | Reads training logs and checks for policy drift, reward regression, waiting-time creep, and training-time spikes. Exits 0/1/2. |
| **Rollback** | `rollback.py` | Save, list, restore, and delete versioned policy snapshots under `models/snapshots/`. |
| **Frontend** | `app.py` | Streamlit + pyvis dashboard with live graph, scoreboard, action log, sidebar controls. |
| **DVC Pipeline** | `dvc.yaml` | Two-stage pipeline (train → evaluate) driven by `params.yaml`. |
| **CI/CD** | `.github/workflows/` | CI runs lint + smoke-train + evaluate + monitoring on every push. CD builds and pushes a Docker image on every merge to `main`. |

---

## Environment Specification

| Property | Value |
|---|---|
| **Graph** | Watts–Strogatz connected graph (n=18, k=4, p=0.3) + extra edges into hub nodes (Database, API_Gateway) |
| **State** | `(current_node_id, time_bucket, tested_bucket, current_tested_flag)` — ~288 discrete states for Q-table lookup |
| **Action** | Index of the neighbour to move to (sorted neighbour list) |
| **Reward** | `+50` critical bug · `+10` minor bug · `-1` per step |
| **Episode end** | All bugs found OR `max_steps` reached |
| **Bug spawn** | Per-node Bernoulli; weight ∈ [0.15, 0.85] depending on module risk; 30% of bugs are critical |

---

## Getting Started

### 1. Install

Pick **either** pip:

```bash
pip install -r requirements.txt
```

**or** conda:

```bash
conda env create -f environment.yml
conda activate bug-hunter-rl
```

### 2. Train the RL Agent

```bash
python train.py --episodes 5000 --seed 42
```

Common flags:

| Flag | Default | Notes |
|---|---|---|
| `--episodes` | `2000` | More episodes → better policy. **5000+ recommended** for a meaningful win over the baseline. |
| `--lr` | `0.1` | Learning rate (alpha) |
| `--gamma` | `0.95` | Discount factor |
| `--epsilon` | `1.0` | Initial exploration rate |
| `--epsilon-decay` | `0.995` | Per-episode decay |
| `--epsilon-min` | `0.05` | Floor on epsilon |
| `--seed` | `42` | Master RNG seed |
| `--n-nodes` | `18` | Graph size (5–18) |
| `--bug-multiplier` | `1.0` | Scales bug-spawn probabilities |
| `--output` | `models/policy_v1.pkl` | Where to save the policy |
| `--mlflow-uri` | *(local file store)* | MLflow tracking URI — omit to use `./mlruns` |

Training logs every run to:
- **MLflow** experiment `bug-hunter-rl` (params, summary metrics, per-100-ep step series, policy artifact)
- **`logs/results.json`** — appended JSON array, one record per run
- **`logs/results.csv`** — flat one-row-per-run CSV

### 3. Evaluate vs Baseline

```bash
python evaluate.py --episodes 100 --seed 42
```

```
========================================================================
 Comparison: Q-Learning Agent vs Random-Walker Baseline
========================================================================
  Metric                       RL Agent               Baseline
  ---------------------------- ---------------------- ----------------------
  Average Reward               152.59 ± 41.20         94.42 ± 60.06
  Average Steps / Episode      24.31                  29.58
  Total Bugs Found             808 / 825              576 / 825
  Bug Discovery Rate           97.94%                 69.82%
========================================================================
  RL improvement over baseline: +57.9% reward
========================================================================
```

Outputs saved:
- `plots/avg_reward_over_episodes.png`
- `plots/bugs_found_over_time.png`
- `logs/eval_metrics.json` (DVC-tracked metrics)
- MLflow experiment `bug-hunter-rl-eval`

### 4. Check System Health

```bash
python monitoring.py
```

Checks performed:
- **Discovery-rate drift** — declining trend across recent training runs
- **Avg waiting-time creep** — steps-to-first-discovery trending upward
- **Reward regression** — latest window reward vs historical peak
- **Training-time spike** — latest run vs rolling average

Exits `0` (GREEN), `1` (WARN with `--strict`), or `2` (CRITICAL). Used as a CI gate.

### 5. Launch the Streamlit Demo

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Use the sidebar to switch agents, tune parameters, and control playback speed. Nodes light up cyan (agent), red (critical bug), orange (minor bug), and grey (tested, no bug).

---

## MLflow Experiment Tracking

MLflow stores all runs locally in `./mlruns` by default — no server required.

```bash
# Launch the MLflow UI to compare runs
mlflow ui
# Open http://localhost:5000
```

To use a remote tracking server:

```bash
python train.py --mlflow-uri http://your-mlflow-server:5000
```

Experiments:
- **`bug-hunter-rl`** — training runs (hyperparams, reward windows, policy artifact)
- **`bug-hunter-rl-eval`** — evaluation runs (RL vs baseline metrics, plots)

---

## DVC Pipeline

`dvc.yaml` defines a reproducible two-stage pipeline. Edit `params.yaml` instead of passing CLI flags, then run `dvc repro`.

```bash
dvc init               # once, after cloning
dvc repro              # run stages that are out-of-date
dvc metrics show       # compare eval_metrics.json across runs
dvc params diff        # see what changed since last run
```

`params.yaml` — all experiment config in one place:

```yaml
train:
  episodes: 2000
  lr: 0.1
  gamma: 0.95
  seed: 42
  ...

evaluate:
  episodes: 100
  seed: 42
  ...
```

To version models with a remote (e.g. S3):

```bash
dvc remote add -d myremote s3://my-bucket/bug-hunter-rl
dvc push   # upload models/policy_v1.pkl
dvc pull   # restore from remote
```

---

## Docker

```bash
# Start the full stack: Streamlit app + MLflow tracking server
docker compose up

# Run an isolated training job
docker compose --profile training up trainer

# Build just the image
docker build -t bug-hunter-rl .
```

| Service | Port | Description |
|---|---|---|
| `app` | 8501 | Streamlit dashboard |
| `mlflow` | 5000 | MLflow tracking UI |
| `trainer` | — | One-off training job (profile: `training`) |

---

## Policy Rollback

```bash
# Snapshot the current policy before an experiment
python rollback.py save --tag before-exp-7

# List all snapshots
python rollback.py list

# Restore a previous snapshot (auto-backs up current first)
python rollback.py restore 20260508_123456_before-exp-7

# Inspect snapshot metadata
python rollback.py info 20260508_123456_before-exp-7
```

Snapshots live in `models/snapshots/` as `.pkl` + `.json` pairs.

---

## CI/CD Pipeline

### CI (on every push / PR)

```
lint (flake8)
    └── smoke-train (100 episodes) + evaluate (20 episodes) + monitoring check
            └── DVC pipeline repro (main / develop only)
```

Artefacts uploaded per run: trained policy, evaluation plots, MLflow run store.

### CD (on merge to `main` or version tag)

Builds and pushes a Docker image to GitHub Container Registry (`ghcr.io`):

```
ghcr.io/<owner>/bug-hunter-rl:latest
ghcr.io/<owner>/bug-hunter-rl:sha-<commit>
ghcr.io/<owner>/bug-hunter-rl:v1.2.3   (on tag)
```

---

## MLOps Logging Schema

Each entry in `logs/results.json`:

```json
{
  "run_id": "16603e5d-c8f8-4e43-b343-b54ca963df0c",
  "timestamp_utc": "2026-05-08T12:49:32+00:00",
  "agent": "QLearningAgent",
  "policy_path": "models/policy_v1.pkl",
  "train_seconds": 14.21,
  "hyperparameters": {
    "episodes": 5000,
    "learning_rate": 0.1,
    "gamma": 0.95,
    "epsilon_initial": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    "seed": 42
  },
  "metrics": {
    "overall_avg_reward": 110.99,
    "final_avg_reward_100": 130.97,
    "bug_discovery_rate": 0.7719,
    "avg_waiting_time": 22.4,
    "bugs_found_total": 38595,
    "bugs_seeded_total": 50000,
    "policy_states_learned": 256
  },
  "windows_per_100": [
    {"episode": 100, "avg_reward_100": 105.02, "epsilon": 0.606},
    "..."
  ]
}
```

`logs/eval_metrics.json` (tracked by DVC):

```json
{
  "rl_avg_reward": 152.59,
  "rl_discovery_rate": 0.9794,
  "baseline_avg_reward": 94.42,
  "baseline_discovery_rate": 0.6982,
  "improvement_pct": 57.9
}
```

---

## Monitoring Plan (Real-World Deployment)

`monitoring.py` implements these checks against live logs:

| Check | WARN threshold | CRITICAL threshold |
|---|---|---|
| Discovery-rate drift | −5% relative over last 5 runs | −15% |
| Avg waiting-time creep | +20% over last 5 runs | +50% |
| Reward regression | < 85% of historical peak | < 70% |
| Training-time spike | > 2× rolling average | > 4× |
| RL vs baseline | < 10% improvement | negative (RL worse) |

---

## SDG 9 Alignment

This project supports **UN Sustainable Development Goal 9 — Industry, Innovation and Infrastructure**:

> Modern digital infrastructure runs on software, and the silent accumulation of bugs in critical services (databases, APIs, payment gateways) is a quiet but serious threat to its resilience. By training an RL agent to autonomously and efficiently traverse a service graph and discover defects, this project demonstrates a scalable mechanism for hardening software systems before they fail in production. Compared with brute-force testing or random fuzzing, an RL bug-hunter learns *which* parts of the codebase are riskiest and concentrates testing effort there — directly supporting SDG 9's goal of building resilient, innovation-friendly infrastructure.

---

## Reproducibility Checklist

- [x] Pinned dependencies (`requirements.txt` and `environment.yml`)
- [x] Master RNG seed flows through env, agent, and bug placement
- [x] Both agents evaluated on identical episode seeds
- [x] Each training run logged with a UUID + UTC timestamp
- [x] Policy serialised to a single pickle with all hyperparameters
- [x] Graph topology is seed-deterministic (only bug placement re-rolls per episode)
- [x] All experiment config in `params.yaml` — single source of truth
- [x] Full pipeline reproducible via `dvc repro`
- [x] CI enforces deterministic smoke-train on every push

---

## Why Tabular Q-Learning (and not DQN)?

The state representation — `(node_id, time_bucket, tested_bucket, tested_flag)` — has roughly **288 discrete states** with our 18-node graph and 4-bucket discretisation. A Q-table fits this comfortably:

- **Fast to train** — full convergence in a few seconds on a laptop.
- **Fully explainable** — you can inspect the Q-table directly and see which neighbour the agent prefers from each state.
- **Trivial to serialise** — a single pickle with no torch/tf dependency.
- **Right tool for the size** — DQN would be over-engineering here and would obscure the explainability story.

If you scale the graph to 100+ nodes or add richer per-node observations, swapping in a DQN would be a clean drop-in (the agent interface already exposes `select_action` / `update`).

---

## Further Reading

- [`docs/requirements.md`](docs/requirements.md) — Full stakeholder analysis, use cases, functional & non-functional requirements, feasibility study, risk analysis, and traceability matrix.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Branching strategy (GitFlow), commit conventions, PR workflow, DVC/MLflow/Docker usage, and release process.
