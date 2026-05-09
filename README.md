#  Autonomous Software Bug Hunter — Reinforcement Learning

> An RL agent that learns to navigate a software-module graph and discover bugs efficiently — with MLOps logging, baseline comparison, and an interactive Streamlit demo.

This project models a microservice-style codebase as a connected graph of modules (Auth, Database, API Gateway, Payment, …) and trains a tabular **Q-Learning** agent to traverse it, test nodes for bugs, and maximise reward while minimising steps. The trained agent is benchmarked against a Random-Walker baseline and visualised live in a web dashboard.

---

##  Project Structure

```
bug_hunter_rl/
├── env.py              # Custom graph environment (no Gym dependency)
├── agent.py            # Q-Learning agent + Random-Walker baseline
├── train.py            # Training loop + MLOps experiment logging
├── evaluate.py         # RL vs baseline comparison + matplotlib plots
├── app.py              # Streamlit interactive dashboard
├── requirements.txt    # pip dependencies
├── environment.yml     # conda environment (alternative)
├── README.md
├── models/             # Saved policies (created by train.py)
├── logs/               # MLOps experiment logs (results.json, results.csv)
└── plots/              # Evaluation plots (created by evaluate.py)
```

---

##  Project Components

| Component | File | Description |
|---|---|---|
| **Simulator** | `env.py` | Connected graph of 18 modules; bugs spawned with per-module risk weights (Database/API/Payment are high-risk). Discrete state, neighbour-indexed actions, reward shaping. |
| **RL Agent** | `agent.py` | Tabular Q-Learning with epsilon-greedy + exponential decay. Saves/loads policy as a pickle. |
| **Baseline Agent** | `agent.py` | Random Walker — picks a uniformly random neighbour every step. |
| **Training + MLOps** | `train.py` | Trains the agent and appends a structured run record (UUID, hyperparameters, metrics, per-100-episode windows) to `logs/results.json` and `logs/results.csv`. |
| **Evaluation** | `evaluate.py` | Runs both agents on identical episode seeds, prints a comparison table, saves two plots, prints SDG-9 alignment text. |
| **Frontend** | `app.py` | Streamlit + pyvis dashboard with live graph, scoreboard, action log, sidebar controls. |

---

##  Environment Specification

| Property | Value |
|---|---|
| **Graph** | Watts–Strogatz connected graph (n=18, k=4, p=0.3) + extra edges into hub nodes (Database, API_Gateway) |
| **State** | `(current_node_id, time_bucket, tested_bucket)` — discretised for Q-table lookup |
| **Action** | Index of the neighbour to move to (sorted neighbour list) |
| **Reward** | `+50` critical bug · `+10` minor bug · `-1` per step |
| **Episode end** | All bugs found OR `max_steps` reached |
| **Bug spawn** | Per-node Bernoulli; weight ∈ [0.15, 0.85] depending on module risk; 30% of bugs are critical |

---

##  Getting Started

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

The script logs:

* **`logs/results.json`** — appended JSON array, one entry per run, with the full hyperparameter dict, per-100-episode reward windows, and final metrics.
* **`logs/results.csv`** — flat one-row-per-run CSV for spreadsheet analysis.

### 3. Evaluate vs Baseline

```bash
python evaluate.py --episodes 100 --seed 42
```

This prints a comparison table to the console, saves two plots, and prints the SDG-9 alignment statement:

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

Plots are saved to `plots/avg_reward_over_episodes.png` and `plots/bugs_found_over_time.png`.

> **Important:** evaluation uses the same `--seed` and `--bug-multiplier` for both agents, so wins are policy-driven, not luck.

### 4. Launch the Streamlit Demo

```bash
streamlit run app.py
```

The browser will open at `http://localhost:8501`. Use the sidebar to:

* Switch between the **RL Agent** and **Random-Walker** baseline
* Point at a specific policy file
* Tune the **bug-spawn multiplier**, **max steps**, and **seed**
* Adjust playback speed for the live simulation

Then click **Run Simulation** to watch the agent traverse the graph in real time. Nodes light up cyan when the agent visits them, red for critical bugs, orange for minor bugs, and grey when tested without finding anything.

---

##  MLOps Logging Schema

Each entry in `logs/results.json` looks like this:

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
    "seed": 42,
    "...": "..."
  },
  "metrics": {
    "overall_avg_reward": 110.99,
    "final_avg_reward_100": 130.97,
    "bug_discovery_rate": 0.7719,
    "bugs_found_total": 38595,
    "bugs_seeded_total": 50000,
    "policy_states_learned": 256
  },
  "windows_per_100": [
    {"episode": 100, "avg_reward_100": 105.02, "epsilon": 0.606},
    ...
  ]
}
```

The CSV log captures the same metrics one-row-per-run for easy comparison across experiments.

---

##  SDG 9 Alignment

This project supports **UN Sustainable Development Goal 9 — Industry, Innovation and Infrastructure**:

> Modern digital infrastructure runs on software, and the silent accumulation of bugs in critical services (databases, APIs, payment gateways) is a quiet but serious threat to its resilience. By training an RL agent to autonomously and efficiently traverse a service graph and discover defects, this project demonstrates a scalable mechanism for hardening software systems before they fail in production. Compared with brute-force testing or random fuzzing, an RL bug-hunter learns *which* parts of the codebase are riskiest and concentrates testing effort there — directly supporting SDG 9's goal of building resilient, innovation-friendly infrastructure.

The same statement is printed by `evaluate.py` and shown in the Streamlit dashboard.

---

##  Reproducibility Checklist

* [x] Pinned dependencies (`requirements.txt` and `environment.yml`)
* [x] Master RNG seed flows through env, agent, and bug placement
* [x] Both agents evaluated on identical episode seeds
* [x] Each training run logged with a UUID + UTC timestamp
* [x] Policy serialised to a single pickle file with all hyperparameters
* [x] Graph topology is seed-deterministic (only bug placement re-rolls per episode)

---

##  Why Tabular Q-Learning (and not DQN)?

The state representation — `(node_id, time_bucket, tested_bucket)` — has roughly **288 discrete states** with our 18-node graph and 4-bucket discretisation. A Q-table fits this comfortably:

* **Fast to train** — full convergence in a few seconds.
* **Fully explainable** — you can inspect the Q-table directly and see which neighbour the agent prefers from each state.
* **Trivial to serialise** — a single pickle with no torch/tf dependency.
* **Right tool for the size of the problem** — DQN would be over-engineering here and would obscure the explainability story.

If you scale the graph to 100+ nodes or add richer per-node observations, swapping in a DQN would be a clean drop-in (the agent interface already exposes `select_action` / `update`).

