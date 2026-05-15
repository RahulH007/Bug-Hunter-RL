# Contributing Guide

## Branching Strategy (GitFlow)

We follow a simplified **GitFlow** model:

```
main        ← production-ready; protected — no direct pushes
develop     ← integration branch; all features merge here first
feature/*   ← new features (branch from develop, PR back to develop)
release/*   ← release stabilisation (branch from develop, PR to main + develop)
hotfix/*    ← emergency production fixes (branch from main, PR to main + develop)
```

### Branch naming conventions

| Prefix | Example | Purpose |
|---|---|---|
| `feature/` | `feature/add-dqn-agent` | New functionality |
| `fix/` | `fix/reward-overflow` | Bug fix in non-production |
| `hotfix/` | `hotfix/numpy-compat` | Urgent fix on `main` |
| `release/` | `release/v1.2.0` | Stabilisation before a release tag |
| `docs/` | `docs/update-readme` | Documentation only |
| `chore/` | `chore/update-deps` | Tooling, CI, dependency bumps |

---

## Commit Message Convention

Follow **Conventional Commits**:

```
<type>(<scope>): <short summary>

[optional body]
[optional footer: Closes #<issue>]
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `perf`

Examples:
```
feat(agent): add Double-Q-Learning variant
fix(env): clip action index to legal neighbours correctly
ci: add flake8 lint step to CI workflow
```

---

## PR Workflow

1. Create a branch from `develop` (or `main` for hotfixes).
2. Make changes; keep commits small and focused.
3. Open a **Draft PR** early for visibility.
4. When ready, move to "Ready for Review" — the PR template auto-fills.
5. CI must pass (`lint` + `train-and-evaluate` jobs) before merge.
6. At least **one approval** required (see `CODEOWNERS`).
7. Squash-merge into `develop`; rebase-merge into `main` for releases.

---

## Development Setup

```bash
# 1. Clone and create your branch
git clone https://github.com/RahulH007/Bug-Hunter-RL.git
cd Bug-Hunter-RL
git checkout -b feature/my-change develop

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Initialise DVC for pipeline versioning
dvc init

# 4. Train a quick test model
python train.py --episodes 200 --quiet

# 5. Evaluate
python evaluate.py --episodes 20

# 6. Check health
python monitoring.py

# 7. Lint before pushing
flake8 *.py --max-line-length=120 --extend-ignore=W503,W504,E203
```

---

## MLflow Experiment Tracking

All training runs are automatically logged to MLflow.

```bash
# Start the MLflow UI (local file store, no server needed)
mlflow ui

# Or start the full stack with Docker Compose
docker compose up

# Point to a remote tracking server
python train.py --mlflow-uri http://your-mlflow-server:5000
```

Experiments:
- **`bug-hunter-rl`** — training runs
- **`bug-hunter-rl-eval`** — evaluation runs

---

## DVC Pipeline

The full train → evaluate pipeline is defined in `dvc.yaml`.
Parameters live in `params.yaml` — edit that file instead of passing CLI flags.

```bash
dvc init              # once, after cloning
dvc repro             # run full pipeline (skips unchanged stages)
dvc metrics show      # compare metrics across runs
dvc params diff       # see what changed since last run
```

To version models with a remote (e.g. S3):
```bash
dvc remote add -d myremote s3://my-bucket/bug-hunter-rl
dvc push              # push models/policy_v1.pkl to remote
dvc pull              # restore from remote
```

---

## Docker

```bash
# Build and run the Streamlit dashboard + MLflow server
docker compose up

# Run a full training job in an isolated container
docker compose --profile training up trainer

# Build just the image
docker build -t bug-hunter-rl .
```

---

## Rollback a Policy

```bash
# Save current policy as a versioned snapshot
python rollback.py save --tag before-experiment-7

# List all snapshots
python rollback.py list

# Restore to a previous snapshot
python rollback.py restore 20260508_123456_before-experiment-7
```

---

## Release Process

1. Create `release/vX.Y.Z` from `develop`.
2. Bump version in README / any version files.
3. Run full test suite and `python monitoring.py --strict`.
4. PR to `main` — squash merge.
5. Tag: `git tag vX.Y.Z && git push --tags`
6. The CD workflow automatically builds and pushes the Docker image.
7. Back-merge `main` → `develop` to keep histories in sync.
