"""
train.py
========
Training driver for the Q-Learning bug-hunting agent.

Run from the project root:
    python train.py --episodes 2000 --lr 0.1 --gamma 0.95
                    --epsilon 1.0 --epsilon-decay 0.995 --seed 42

What it does
------------
1. Builds the BugHuntingEnv and a QLearningAgent.
2. Runs the requested number of training episodes.
3. Logs per-episode metrics in memory and aggregates them every 100 episodes.
4. Persists:
   * the trained policy ........... -> models/policy_v1.pkl
   * a JSON record of this run ..... -> logs/results.json   (appended)
   * a CSV row for spreadsheet use . -> logs/results.csv    (appended)

The MLOps log captures: run-id (UUID + timestamp), all hyperparameters,
total episodes, average reward per 100 episodes, and bug-discovery rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from agent import QLearningAgent
from env import BugHuntingEnv


ROOT       = Path(__file__).parent.resolve()
MODEL_DIR  = ROOT / "models"
LOG_DIR    = ROOT / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = MODEL_DIR / "policy_v1.pkl"
JSON_LOG_PATH      = LOG_DIR / "results.json"
CSV_LOG_PATH       = LOG_DIR / "results.csv"


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Q-learning bug-hunting agent.",
    )
    p.add_argument("--episodes",      type=int,   default=2000,
                   help="Number of training episodes.")
    p.add_argument("--max-steps",     type=int,   default=30,
                   help="Max steps per episode.")
    p.add_argument("--n-nodes",       type=int,   default=18,
                   help="Number of nodes in the software graph (5-18).")
    p.add_argument("--lr",            type=float, default=0.1,
                   help="Learning rate (alpha).")
    p.add_argument("--gamma",         type=float, default=0.95,
                   help="Discount factor.")
    p.add_argument("--epsilon",       type=float, default=1.0,
                   help="Initial exploration rate.")
    p.add_argument("--epsilon-min",   type=float, default=0.05,
                   help="Floor on epsilon.")
    p.add_argument("--epsilon-decay", type=float, default=0.995,
                   help="Per-episode multiplicative decay for epsilon.")
    p.add_argument("--bug-multiplier", type=float, default=1.0,
                   help="Global multiplier on per-node bug spawn probabilities.")
    p.add_argument("--seed",          type=int,   default=42,
                   help="Master RNG seed.")
    p.add_argument("--output",        type=str,   default=str(DEFAULT_MODEL_PATH),
                   help="Where to save the trained policy pickle.")
    p.add_argument("--quiet",         action="store_true",
                   help="Suppress per-window progress output.")
    return p.parse_args()



def train(args: argparse.Namespace) -> Dict:
    """
    Run training and return a dict of summary metrics suitable for logging.
    """
    env = BugHuntingEnv(
        n_nodes              = args.n_nodes,
        max_steps            = args.max_steps,
        bug_spawn_multiplier = args.bug_multiplier,
        seed                 = args.seed,
    )
    agent = QLearningAgent(
        n_actions       = env.max_degree,
        learning_rate   = args.lr,
        discount_factor = args.gamma,
        epsilon         = args.epsilon,
        epsilon_min     = args.epsilon_min,
        epsilon_decay   = args.epsilon_decay,
        seed            = args.seed,
    )

    rewards_per_episode: List[float] = []
    bugs_per_episode:    List[int]   = []
    steps_per_episode:   List[int]   = []
    bugs_seeded_total:   int         = 0
    bugs_found_total:    int         = 0
    window_means:        List[Dict]  = []   # one entry per 100-episode window

    t0 = time.time()
    for ep in range(1, args.episodes + 1):
        # We re-seed each episode with a deterministic-but-varying seed so
        # bug placements are different across episodes but reproducible
        # across runs (useful for fair comparison in evaluate.py).
        state = env.reset(seed=args.seed + ep)
        legal = env.get_legal_actions()
        ep_reward = 0.0

        while True:
            action = agent.select_action(state, legal, greedy=False)
            out = env.step(action)
            next_legal = env.get_legal_actions()

            agent.update(
                state              = state,
                action             = action,
                reward             = out.reward,
                next_state         = out.state,
                done               = out.done,
                next_legal_actions = next_legal,
            )

            state    = out.state
            legal    = next_legal
            ep_reward += out.reward
            if out.done:
                break

        agent.decay_epsilon()
        rewards_per_episode.append(ep_reward)
        bugs_per_episode.append(len(env.found_bugs))
        steps_per_episode.append(env.steps_taken)
        bugs_seeded_total += env.total_bugs_at_start
        bugs_found_total  += len(env.found_bugs)

        # Windowed reporting every 100 eps — handy for sanity-checking
        # learning is actually happening during a long run.
        if ep % 100 == 0:
            window_avg_r = float(np.mean(rewards_per_episode[-100:]))
            window_avg_b = float(np.mean(bugs_per_episode[-100:]))
            window_means.append({
                "episode":        ep,
                "avg_reward_100": round(window_avg_r, 3),
                "avg_bugs_100":   round(window_avg_b, 3),
                "epsilon":        round(agent.epsilon, 4),
            })
            if not args.quiet:
                print(
                    f"[ep {ep:>5}/{args.episodes}] "
                    f"avg_reward(100) = {window_avg_r:7.2f}  "
                    f"avg_bugs(100) = {window_avg_b:5.2f}  "
                    f"epsilon = {agent.epsilon:.3f}"
                )

    train_time = time.time() - t0


    out_path = Path(args.output)
    agent.save(out_path)
    if not args.quiet:
        print(f"\n[OK] Saved policy to {out_path} "
              f"({agent.policy_size()} states with non-zero Q-values)")


    overall_avg_reward = float(np.mean(rewards_per_episode))
    bug_discovery_rate = (bugs_found_total / bugs_seeded_total
                          if bugs_seeded_total else 0.0)

    run_record = {
        "run_id":          str(uuid.uuid4()),
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent":           "QLearningAgent",
        "policy_path":     str(out_path),
        "train_seconds":   round(train_time, 2),
        "hyperparameters": {
            "episodes":             args.episodes,
            "max_steps":            args.max_steps,
            "n_nodes":              args.n_nodes,
            "learning_rate":        args.lr,
            "gamma":                args.gamma,
            "epsilon_initial":      args.epsilon,
            "epsilon_min":          args.epsilon_min,
            "epsilon_decay":        args.epsilon_decay,
            "bug_spawn_multiplier": args.bug_multiplier,
            "seed":                 args.seed,
        },
        "metrics": {
            "total_episodes":         args.episodes,
            "overall_avg_reward":     round(overall_avg_reward, 3),
            "final_avg_reward_100":   (window_means[-1]["avg_reward_100"]
                                       if window_means else None),
            "first_avg_reward_100":   (window_means[0]["avg_reward_100"]
                                       if window_means else None),
            "bug_discovery_rate":     round(bug_discovery_rate, 4),
            "avg_waiting_time":       round(float(np.mean(steps_per_episode)), 2),
            "bugs_found_total":       bugs_found_total,
            "bugs_seeded_total":      bugs_seeded_total,
            "policy_states_learned":  agent.policy_size(),
        },
        "windows_per_100": window_means,
    }
    return run_record



def append_json_log(record: Dict, path: Path = JSON_LOG_PATH) -> None:
    """
    Append the run record to a JSON-array log file. We re-write the file
    each call (small file, simple) — keeps it valid JSON for downstream
    tooling instead of using JSON-Lines.
    """
    history: List[Dict] = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = []
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def append_csv_log(record: Dict, path: Path = CSV_LOG_PATH) -> None:
    """Flatten the record into a single CSV row for spreadsheet analysis."""
    hp = record["hyperparameters"]
    m  = record["metrics"]
    row = {
        "run_id":              record["run_id"],
        "timestamp_utc":       record["timestamp_utc"],
        "agent":               record["agent"],
        "episodes":            hp["episodes"],
        "learning_rate":       hp["learning_rate"],
        "gamma":               hp["gamma"],
        "epsilon_initial":     hp["epsilon_initial"],
        "epsilon_decay":       hp["epsilon_decay"],
        "epsilon_min":         hp["epsilon_min"],
        "seed":                hp["seed"],
        "n_nodes":             hp["n_nodes"],
        "bug_multiplier":      hp["bug_spawn_multiplier"],
        "overall_avg_reward":  m["overall_avg_reward"],
        "final_avg_reward_100":m["final_avg_reward_100"],
        "bug_discovery_rate":  m["bug_discovery_rate"],
        "avg_waiting_time":    m["avg_waiting_time"],
        "bugs_found_total":    m["bugs_found_total"],
        "bugs_seeded_total":   m["bugs_seeded_total"],
        "train_seconds":       record["train_seconds"],
    }
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    print("=" * 64)
    print(" Bug-Hunter RL — Training Run")
    print("=" * 64)
    for k, v in vars(args).items():
        print(f"  {k:<16}: {v}")
    print("-" * 64)

    record = train(args)

    append_json_log(record)
    append_csv_log(record)

    print("-" * 64)
    print(f"  Run ID            : {record['run_id']}")
    print(f"  Avg reward overall: {record['metrics']['overall_avg_reward']}")
    print(f"  Final 100-window R: {record['metrics']['final_avg_reward_100']}")
    print(f"  Bug discovery rate: {record['metrics']['bug_discovery_rate']:.2%}")
    print(f"  Logged to         : {JSON_LOG_PATH}")
    print(f"                      {CSV_LOG_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
