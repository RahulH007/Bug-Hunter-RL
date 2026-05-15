"""
monitoring.py
=============
Operational health monitor for the Bug-Hunter RL system.

Reads logs/results.json (training history) and logs/eval_metrics.json
(latest evaluation snapshot) to detect:

  1. Bug-discovery-rate DRIFT  — declining trend across the last N training runs.
  2. Avg-waiting-time CREEP    — increasing steps-to-first-discovery trend.
  3. Reward REGRESSION         — final window reward below a % of the peak seen.
  4. Training-time SPIKE       — latest run took >2x the rolling average.

Exit codes
----------
  0  All checks GREEN
  1  At least one WARNING
  2  At least one CRITICAL alert

Usage
-----
    python monitoring.py [--log logs/results.json] [--window 5] [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Thresholds ────────────────────────────────────────────────────────────────

DRIFT_WARN_THRESHOLD      = 0.05   # 5% relative drop in discovery rate → WARN
DRIFT_CRITICAL_THRESHOLD  = 0.15   # 15% relative drop                  → CRITICAL

WAIT_WARN_THRESHOLD       = 0.20   # 20% increase in avg_waiting_time    → WARN
WAIT_CRITICAL_THRESHOLD   = 0.50   # 50% increase                        → CRITICAL

REWARD_WARN_PCT           = 0.85   # final window reward < 85% of peak   → WARN
REWARD_CRITICAL_PCT       = 0.70   # final window reward < 70% of peak   → CRITICAL

TRAIN_TIME_WARN_MULT      = 2.0    # latest run > 2x rolling avg         → WARN
TRAIN_TIME_CRITICAL_MULT  = 4.0    # latest run > 4x rolling avg         → CRITICAL


# ── ANSI colour helpers ───────────────────────────────────────────────────────

def _green(s: str)  -> str: return f"\033[92m{s}\033[0m"
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m"
def _red(s: str)    -> str: return f"\033[91m{s}\033[0m"
def _bold(s: str)   -> str: return f"\033[1m{s}\033[0m"


# ── Core checks ───────────────────────────────────────────────────────────────

def _trend(values: List[float]) -> float:
    """Return (last - first) / abs(first); positive = growing, negative = shrinking."""
    if len(values) < 2 or abs(values[0]) < 1e-9:
        return 0.0
    return (values[-1] - values[0]) / abs(values[0])


def check_discovery_rate_drift(
    runs: List[Dict], window: int
) -> Tuple[str, str]:
    """CRITICAL / WARN if bug discovery rate is declining over the last `window` runs."""
    rates = [r["metrics"]["bug_discovery_rate"] for r in runs[-window:] if "metrics" in r]
    if len(rates) < 2:
        return "GREEN", "Not enough runs to assess drift (need ≥ 2)."
    change = _trend(rates)
    if change <= -DRIFT_CRITICAL_THRESHOLD:
        return "CRITICAL", (
            f"Bug discovery rate dropped {abs(change):.1%} over last {len(rates)} runs "
            f"({rates[0]:.2%} → {rates[-1]:.2%}). Model may need retraining."
        )
    if change <= -DRIFT_WARN_THRESHOLD:
        return "WARN", (
            f"Bug discovery rate declining {abs(change):.1%} over last {len(rates)} runs "
            f"({rates[0]:.2%} → {rates[-1]:.2%})."
        )
    return "GREEN", (
        f"Bug discovery rate stable/improving over last {len(rates)} runs "
        f"({rates[0]:.2%} → {rates[-1]:.2%})."
    )


def check_waiting_time_creep(
    runs: List[Dict], window: int
) -> Tuple[str, str]:
    """CRITICAL / WARN if average steps-to-discovery is trending upward."""
    times = [
        r["metrics"]["avg_waiting_time"]
        for r in runs[-window:]
        if "metrics" in r and "avg_waiting_time" in r["metrics"]
    ]
    if len(times) < 2:
        return "GREEN", "Not enough runs to assess waiting-time trend (need ≥ 2)."
    change = _trend(times)
    if change >= WAIT_CRITICAL_THRESHOLD:
        return "CRITICAL", (
            f"Avg waiting time increased {change:.1%} over last {len(times)} runs "
            f"({times[0]:.1f} → {times[-1]:.1f} steps). Agent efficiency degrading."
        )
    if change >= WAIT_WARN_THRESHOLD:
        return "WARN", (
            f"Avg waiting time up {change:.1%} over last {len(times)} runs "
            f"({times[0]:.1f} → {times[-1]:.1f} steps)."
        )
    return "GREEN", (
        f"Avg waiting time stable/improving ({times[0]:.1f} → {times[-1]:.1f} steps)."
    )


def check_reward_regression(
    runs: List[Dict], window: int
) -> Tuple[str, str]:
    """CRITICAL / WARN if the latest run's final window reward is far below peak."""
    rewards = [
        r["metrics"].get("final_avg_reward_100")
        for r in runs
        if "metrics" in r and r["metrics"].get("final_avg_reward_100") is not None
    ]
    if len(rewards) < 2:
        return "GREEN", "Not enough data to assess reward regression (need ≥ 2 runs)."
    peak    = max(rewards)
    latest  = rewards[-1]
    ratio   = latest / peak if peak > 0 else 1.0
    if ratio < REWARD_CRITICAL_PCT:
        return "CRITICAL", (
            f"Latest final window reward ({latest:.1f}) is {ratio:.0%} of peak ({peak:.1f}). "
            f"Significant policy regression detected."
        )
    if ratio < REWARD_WARN_PCT:
        return "WARN", (
            f"Latest final window reward ({latest:.1f}) is {ratio:.0%} of peak ({peak:.1f})."
        )
    return "GREEN", (
        f"Final window reward healthy: {latest:.1f} ({ratio:.0%} of peak {peak:.1f})."
    )


def check_training_time_spike(
    runs: List[Dict], window: int
) -> Tuple[str, str]:
    """CRITICAL / WARN if the latest run took much longer than normal."""
    times = [r.get("train_seconds", 0) for r in runs if r.get("train_seconds")]
    if len(times) < 2:
        return "GREEN", "Not enough runs to assess training-time baseline (need ≥ 2)."
    historical_avg = sum(times[:-1]) / len(times[:-1])
    latest         = times[-1]
    if historical_avg < 1e-3:
        return "GREEN", "Historical training time near-zero — skipping spike check."
    ratio = latest / historical_avg
    if ratio >= TRAIN_TIME_CRITICAL_MULT:
        return "CRITICAL", (
            f"Latest training took {latest:.1f}s — {ratio:.1f}x the historical avg "
            f"({historical_avg:.1f}s). Possible resource contention."
        )
    if ratio >= TRAIN_TIME_WARN_MULT:
        return "WARN", (
            f"Latest training took {latest:.1f}s — {ratio:.1f}x the historical avg "
            f"({historical_avg:.1f}s)."
        )
    return "GREEN", f"Training time nominal: {latest:.1f}s (avg {historical_avg:.1f}s)."


def check_eval_metrics(eval_path: Path) -> Tuple[str, str]:
    """WARN if eval_metrics.json shows RL is barely beating the baseline."""
    if not eval_path.exists():
        return "GREEN", "No eval_metrics.json found — skipping evaluation check."
    with open(eval_path, encoding="utf-8") as f:
        m = json.load(f)
    improvement = m.get("improvement_pct", None)
    if improvement is None:
        return "GREEN", "improvement_pct not found in eval_metrics.json."
    if improvement < 0:
        return "CRITICAL", (
            f"RL agent is WORSE than random baseline by {abs(improvement):.1f}% reward."
        )
    if improvement < 10:
        return "WARN", (
            f"RL agent only {improvement:.1f}% better than random — consider more training."
        )
    return "GREEN", f"RL agent outperforms baseline by {improvement:.1f}%."


# ── Report helpers ────────────────────────────────────────────────────────────

_LEVEL_ORDER = {"GREEN": 0, "WARN": 1, "CRITICAL": 2}


def _fmt(level: str, label: str, message: str) -> str:
    icon    = {"GREEN": "OK  ", "WARN": "WARN", "CRITICAL": "CRIT"}[level]
    colourf = {"GREEN": _green, "WARN": _yellow, "CRITICAL": _red}[level]
    return f"  [{colourf(icon)}] {_bold(label)}: {message}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Health monitor for Bug-Hunter RL.")
    p.add_argument("--log",    default="logs/results.json",
                   help="Path to MLOps training log (default: logs/results.json).")
    p.add_argument("--eval",   default="logs/eval_metrics.json",
                   help="Path to evaluation metrics JSON.")
    p.add_argument("--window", type=int, default=5,
                   help="Number of recent runs to use for trend analysis.")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 on any WARN (useful for CI hard gates).")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    log_path  = Path(args.log)
    eval_path = Path(args.eval)

    print(_bold("=" * 60))
    print(_bold("  Bug-Hunter RL — Operational Health Report"))
    print(_bold("=" * 60))

    # Load training history
    runs: List[Dict] = []
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                data = json.load(f)
            runs = data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            runs = []

    if not runs:
        print(_yellow("  No training runs found in log. Run `python train.py` first."))
        print()
        sys.exit(0)

    print(f"  Analysing {len(runs)} training run(s) "
          f"(window = last {min(args.window, len(runs))}).")
    print()

    results = [
        ("Discovery-rate drift",   check_discovery_rate_drift(runs, args.window)),
        ("Avg waiting-time creep", check_waiting_time_creep(runs, args.window)),
        ("Reward regression",      check_reward_regression(runs, args.window)),
        ("Training-time spike",    check_training_time_spike(runs, args.window)),
        ("Eval vs baseline",       check_eval_metrics(eval_path)),
    ]

    worst = "GREEN"
    for label, (level, message) in results:
        print(_fmt(level, label, message))
        if _LEVEL_ORDER[level] > _LEVEL_ORDER[worst]:
            worst = level

    print()
    print(_bold("=" * 60))
    overall_colour = {"GREEN": _green, "WARN": _yellow, "CRITICAL": _red}[worst]
    print(f"  Overall status: {overall_colour(_bold(worst))}")
    print(_bold("=" * 60))

    if worst == "CRITICAL":
        sys.exit(2)
    if worst == "WARN" and args.strict:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
