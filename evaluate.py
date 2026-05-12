"""
evaluate.py
===========
Head-to-head evaluation of the trained Q-Learning agent vs the
Random-Walker baseline.

Both agents are run on the *same* set of episode seeds, so any difference
in performance is attributable to the policy alone — not luckier bug
placements.

Outputs
-------
1. A console comparison table (Average Reward, Average Steps, Total Bugs).
2. A short SDG-9 alignment statement printed to console.
3. Two matplotlib figures saved under plots/:
    * plots/avg_reward_over_episodes.png
    * plots/bugs_found_over_time.png

Usage
-----
    python evaluate.py --episodes 100 --policy models/policy_v1.pkl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Use a non-interactive backend so the script works on headless servers
# (CI, evaluators' machines without a display) — figures still save fine.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent, RandomWalkerAgent
from env import BugHuntingEnv


ROOT      = Path(__file__).parent.resolve()
PLOT_DIR  = ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_POLICY = ROOT / "models" / "policy_v1.pkl"


SDG_TEXT = (
    "SDG 9 — Industry, Innovation and Infrastructure\n"
    "------------------------------------------------\n"
    "Modern digital infrastructure runs on software, and the silent\n"
    "accumulation of bugs in critical services (databases, APIs, payment\n"
    "gateways) is a quiet but serious threat to its resilience. By\n"
    "training an RL agent to autonomously and efficiently traverse a\n"
    "service graph and discover defects, this project demonstrates a\n"
    "scalable mechanism for hardening software systems before they fail\n"
    "in production. Compared with brute-force testing or random fuzzing,\n"
    "an RL bug-hunter learns *which* parts of the codebase are riskiest\n"
    "and concentrates testing effort there — directly supporting SDG 9's\n"
    "goal of building resilient, innovation-friendly infrastructure."
)



@dataclass
class EpisodeStats:
    reward:        float
    steps:         int
    bugs_found:    int
    bugs_seeded:   int


def run_episode(
    env,
    agent,
    episode_seed: int,
    greedy: bool = True,
) -> EpisodeStats:
    """Run one episode of the env with the given agent and return its stats."""
    state = env.reset(seed=episode_seed)
    legal = env.get_legal_actions()
    total_reward = 0.0

    while True:
        action = agent.select_action(state, legal, greedy=greedy)
        out = env.step(action)
        total_reward += out.reward
        state = out.state
        legal = env.get_legal_actions()
        if out.done:
            break

    return EpisodeStats(
        reward      = total_reward,
        steps       = env.steps_taken,
        bugs_found  = len(env.found_bugs),
        bugs_seeded = env.total_bugs_at_start,
    )


def evaluate_agents(
    n_episodes:   int,
    policy_path:  Path,
    n_nodes:      int,
    max_steps:    int,
    bug_mult:     float,
    base_seed:    int,
) -> Tuple[List[EpisodeStats], List[EpisodeStats]]:
    """
    Run both agents on identical episode seeds.

    Returns
    -------
    (rl_stats, baseline_stats) — each a list of length n_episodes.
    """
    # Two independent env instances, but identical configuration & seed,
    # ensure we exercise *exactly* the same bug placements per episode.
    env_rl   = BugHuntingEnv(n_nodes=n_nodes, max_steps=max_steps,
                              bug_spawn_multiplier=bug_mult, seed=base_seed)
    env_base = BugHuntingEnv(n_nodes=n_nodes, max_steps=max_steps,
                              bug_spawn_multiplier=bug_mult, seed=base_seed)

    rl_agent = QLearningAgent.load(policy_path)
    baseline = RandomWalkerAgent(n_actions=env_base.max_degree, seed=base_seed)

    rl_stats:   List[EpisodeStats] = []
    base_stats: List[EpisodeStats] = []

    for ep in range(n_episodes):
        episode_seed = base_seed + 100_000 + ep   # fresh seed range; same for both
        rl_stats.append(run_episode(env_rl,   rl_agent, episode_seed, greedy=True))
        base_stats.append(run_episode(env_base, baseline, episode_seed, greedy=False))

    return rl_stats, base_stats



def summarise(stats: List[EpisodeStats]) -> Dict[str, float]:
    rewards = np.array([s.reward for s in stats], dtype=float)
    steps   = np.array([s.steps  for s in stats], dtype=float)
    found   = np.array([s.bugs_found  for s in stats], dtype=float)
    seeded  = np.array([s.bugs_seeded for s in stats], dtype=float)
    discovery_rate = (found.sum() / seeded.sum()) if seeded.sum() > 0 else 0.0
    return {
        "avg_reward":     float(rewards.mean()),
        "std_reward":     float(rewards.std()),
        "avg_steps":      float(steps.mean()),
        "total_bugs":     int(found.sum()),
        "seeded_bugs":    int(seeded.sum()),
        "discovery_rate": float(discovery_rate),
    }


def print_comparison_table(
    rl_summary:   Dict[str, float],
    base_summary: Dict[str, float],
) -> None:
    """Pretty-print a side-by-side comparison to stdout."""
    rows = [
        ("Average Reward",
         f"{rl_summary['avg_reward']:.2f} ± {rl_summary['std_reward']:.2f}",
         f"{base_summary['avg_reward']:.2f} ± {base_summary['std_reward']:.2f}"),
        ("Average Steps / Episode",
         f"{rl_summary['avg_steps']:.2f}",
         f"{base_summary['avg_steps']:.2f}"),
        ("Total Bugs Found",
         f"{rl_summary['total_bugs']} / {rl_summary['seeded_bugs']}",
         f"{base_summary['total_bugs']} / {base_summary['seeded_bugs']}"),
        ("Bug Discovery Rate",
         f"{rl_summary['discovery_rate']:.2%}",
         f"{base_summary['discovery_rate']:.2%}"),
    ]
    print()
    print("=" * 72)
    print(" Comparison: Q-Learning Agent vs Random-Walker Baseline")
    print("=" * 72)
    print(f"  {'Metric':<28} {'RL Agent':<22} {'Baseline':<22}")
    print(f"  {'-'*28} {'-'*22} {'-'*22}")
    for name, rl_val, base_val in rows:
        print(f"  {name:<28} {rl_val:<22} {base_val:<22}")
    print("=" * 72)

    # Headline takeaway
    delta_r = rl_summary["avg_reward"] - base_summary["avg_reward"]
    sign    = "+" if delta_r >= 0 else ""
    pct     = (
        100.0 * delta_r / abs(base_summary["avg_reward"])
        if base_summary["avg_reward"] != 0 else float("nan")
    )
    print(f"  RL improvement over baseline: {sign}{delta_r:.2f} reward "
          f"({sign}{pct:.1f}%)")
    print("=" * 72)


def plot_average_reward(
    rl_stats:   List[EpisodeStats],
    base_stats: List[EpisodeStats],
    out_path:   Path,
) -> None:
    """Plot per-episode reward + a rolling mean for both agents."""
    rl_rewards   = np.array([s.reward for s in rl_stats])
    base_rewards = np.array([s.reward for s in base_stats])
    episodes     = np.arange(1, len(rl_stats) + 1)

    # Rolling window: clamp to a sensible size relative to total length.
    window = max(5, min(20, len(rl_stats) // 5))
    def rolling(x: np.ndarray) -> np.ndarray:
        if len(x) < window:
            return x
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, rl_rewards,   alpha=0.25, color="C0", label="RL (per ep)")
    ax.plot(episodes, base_rewards, alpha=0.25, color="C1", label="Baseline (per ep)")

    rl_roll   = rolling(rl_rewards)
    base_roll = rolling(base_rewards)
    roll_x    = episodes[len(episodes) - len(rl_roll):]
    ax.plot(roll_x, rl_roll,   color="C0", linewidth=2.4, label=f"RL rolling-{window}")
    ax.plot(roll_x, base_roll, color="C1", linewidth=2.4, label=f"Baseline rolling-{window}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Average reward over evaluation episodes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bugs_over_time(
    rl_stats:   List[EpisodeStats],
    base_stats: List[EpisodeStats],
    out_path:   Path,
) -> None:
    """
    Plot cumulative bugs found across episodes — gives a "bugs found over
    time" curve since each episode is a unit of wall-clock testing effort.
    """
    rl_cum   = np.cumsum([s.bugs_found for s in rl_stats])
    base_cum = np.cumsum([s.bugs_found for s in base_stats])
    seed_cum = np.cumsum([s.bugs_seeded for s in rl_stats])  # same for both
    episodes = np.arange(1, len(rl_stats) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, seed_cum, color="grey", linestyle="--",
            label="Bugs seeded (oracle ceiling)")
    ax.plot(episodes, rl_cum,   color="C0", linewidth=2.4, label="RL agent")
    ax.plot(episodes, base_cum, color="C1", linewidth=2.4, label="Baseline")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative bugs found")
    ax.set_title("Bugs found over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RL agent vs baseline.")
    p.add_argument("--episodes", type=int,   default=100,
                   help="Number of evaluation episodes per agent.")
    p.add_argument("--policy",   type=str,   default=str(DEFAULT_POLICY),
                   help="Path to the trained Q-learning policy pickle.")
    p.add_argument("--n-nodes",  type=int,   default=18)
    p.add_argument("--max-steps", type=int,  default=30)
    p.add_argument("--bug-multiplier", type=float, default=1.0)
    p.add_argument("--seed",     type=int,   default=42,
                   help="Base seed — must match the env config used in training "
                        "for like-for-like comparison.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Policy file not found: {policy_path}\n"
            f"  -> Run `python train.py` first."
        )

    print(f"Evaluating over {args.episodes} episodes "
          f"(policy: {policy_path.name}, seed: {args.seed}) ...")

    rl_stats, base_stats = evaluate_agents(
        n_episodes  = args.episodes,
        policy_path = policy_path,
        n_nodes     = args.n_nodes,
        max_steps   = args.max_steps,
        bug_mult    = args.bug_multiplier,
        base_seed   = args.seed,
    )

    rl_summary   = summarise(rl_stats)
    base_summary = summarise(base_stats)
    print_comparison_table(rl_summary, base_summary)

    # Plots
    reward_plot = PLOT_DIR / "avg_reward_over_episodes.png"
    bugs_plot   = PLOT_DIR / "bugs_found_over_time.png"
    plot_average_reward(rl_stats, base_stats, reward_plot)
    plot_bugs_over_time(rl_stats, base_stats, bugs_plot)
    print(f"  Saved plot: {reward_plot}")
    print(f"  Saved plot: {bugs_plot}")

    # SDG-9 alignment statement
    print()
    print(SDG_TEXT)
    print()


if __name__ == "__main__":
    main()
