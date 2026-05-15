## Summary

<!-- 1-3 bullet points describing what this PR does and why -->

-
-

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / code cleanup
- [ ] Documentation update
- [ ] CI/CD / tooling change
- [ ] MLOps / experiment tracking change

## Testing

- [ ] Ran `python train.py --episodes 100 --quiet` successfully
- [ ] Ran `python evaluate.py --episodes 20` successfully
- [ ] Ran `python monitoring.py` — no CRITICAL alerts
- [ ] Ran `flake8` with no new errors

## MLflow / DVC

- [ ] Any new hyperparameters are logged via `mlflow.log_params()`
- [ ] Any new metrics are logged via `mlflow.log_metrics()`
- [ ] `params.yaml` updated if training config changed
- [ ] `dvc.yaml` updated if pipeline stages changed

## Checklist

- [ ] Branch follows naming convention (`feature/`, `hotfix/`, `release/`)
- [ ] Commit messages are clear and reference the change
- [ ] No secrets, credentials, or `.env` files committed
- [ ] `requirements.txt` updated if new dependencies added
