"""
agent.py
========
Reinforcement-learning agents for the Bug-Hunting environment.

This module exposes two agents that share a common interface
(`select_action`, `update`, `save`, `load`) so the training and evaluation
scripts can swap them transparently:

* `QLearningAgent`  — tabular Q-learning with epsilon-greedy exploration
                      and exponential epsilon decay.
* `RandomWalkerAgent` — uniform-random baseline that ignores rewards.

Why tabular Q-learning?
-----------------------
The environment exposes a small, discrete state tuple
(current_node, time_bucket, tested_bucket). With ~18 nodes × 4 time
buckets × 4 tested buckets ≈ 288 states and a small action space, a Q-table
is fast to learn, trivially serialisable, and — crucially for a university
evaluation — *fully explainable*. We can print the Q-table and reason about
why the agent prefers certain transitions, which a deep network would
hide behind weights.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


State = Tuple[int, int, int, int]   # (node_id, time_bucket, tested_bucket, current_tested_flag)


class QLearningAgent:
    """
    Tabular Q-learning with epsilon-greedy exploration.

    Update rule (standard Bellman):
        Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_actions : int
            Size of the action space (= env.max_degree). Larger than the
            average node degree, so we'll often have "unused" action slots
            — that's fine; the env clips invalid actions modulo
            len(legal_actions).
        learning_rate : float
            Alpha — step size for Q-table updates.
        discount_factor : float
            Gamma — how much future reward is valued vs immediate reward.
        epsilon : float
            Starting exploration rate (probability of random action).
        epsilon_min : float
            Floor on epsilon — we never stop exploring entirely.
        epsilon_decay : float
            Multiplicative decay applied to epsilon after each episode.
        seed : Optional[int]
            RNG seed for reproducibility of the exploration policy.
        """
        self.n_actions       = n_actions
        self.lr              = learning_rate
        self.gamma           = discount_factor
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.epsilon_initial = epsilon

        # Q-table: defaultdict so unseen states return a zero-vector.
        # We store as np.ndarray for fast argmax / max operations.
        self.q_table: Dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)


    def select_action(
        self,
        state: State,
        legal_actions: List[int],
        greedy: bool = False,
    ) -> int:
        """
        Pick an action index in [0, len(legal_actions)).

        Note: we return an *index into the legal_actions list*, not a raw
        node id. This keeps the action space stable across states (some
        nodes have 4 neighbours, others 9) — the Q-table dimension is
        n_actions = max_degree, and we mask to the legal subset.

        Parameters
        ----------
        greedy : bool
            If True, ignore epsilon and always pick the best-known action.
            Used during evaluation.
        """
        n_legal = len(legal_actions)
        if n_legal == 0:
            return 0

        # Epsilon-greedy: with prob epsilon, explore uniformly at random.
        if (not greedy) and self._rng.random() < self.epsilon:
            return self._rng.randrange(n_legal)

        # Greedy: pick the legal action with the highest Q-value.
        # We slice the Q-vector to the first n_legal entries because the
        # env clips actions modulo len(legal_actions). Any tie is broken
        # randomly to avoid pathological deterministic loops.
        q_values = self.q_table[state][:n_legal]
        max_q    = q_values.max()
        best     = [i for i, q in enumerate(q_values) if q == max_q]
        return self._rng.choice(best)


    def update(
        self,
        state:      State,
        action:     int,
        reward:     float,
        next_state: State,
        done:       bool,
        next_legal_actions: Optional[List[int]] = None,
    ) -> None:
        """
        Apply one Bellman update. The `next_legal_actions` parameter lets us
        bootstrap only off legal next-step Q-values, which avoids the agent
        learning fictitious value from out-of-range action slots.
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            if next_legal_actions:
                next_q_slice = self.q_table[next_state][: len(next_legal_actions)]
                next_max = next_q_slice.max() if len(next_q_slice) > 0 else 0.0
            else:
                next_max = self.q_table[next_state].max()
            target = reward + self.gamma * next_max

        # Standard Q-learning update
        self.q_table[state][action] = current_q + self.lr * (target - current_q)

    def decay_epsilon(self) -> None:
        """Call once per episode to anneal exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def save(self, path: str | Path) -> None:
        """Pickle the Q-table and hyperparameters to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert defaultdict -> dict for clean serialization.
        payload = {
            "q_table": dict(self.q_table),
            "n_actions":     self.n_actions,
            "lr":            self.lr,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        """Reconstruct an agent from a saved policy file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = cls(
            n_actions      = payload["n_actions"],
            learning_rate  = payload["lr"],
            discount_factor= payload["gamma"],
            epsilon        = payload["epsilon"],
            epsilon_min    = payload["epsilon_min"],
            epsilon_decay  = payload["epsilon_decay"],
        )
        # Re-wrap stored dict back into a defaultdict so unseen states
        # still return a zero-vector at inference time.
        for k, v in payload["q_table"].items():
            agent.q_table[k] = np.asarray(v, dtype=np.float64)
        return agent


    def policy_size(self) -> int:
        """Number of states for which we have non-trivial Q-values."""
        return sum(1 for v in self.q_table.values() if np.any(v != 0))


class RandomWalkerAgent:
    """
    A trivial baseline: at every state, pick a uniformly random legal action.
    Ignores rewards entirely — used to quantify the value added by RL.
    """

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        self.n_actions = n_actions
        self._rng = random.Random(seed)

    def select_action(
        self,
        state: State,
        legal_actions: List[int],
        greedy: bool = False,  # ignored — random walker has no policy
    ) -> int:
        n_legal = len(legal_actions)
        return 0 if n_legal == 0 else self._rng.randrange(n_legal)

    # No-op stubs so the training/evaluation loops can call them uniformly.
    def update(self, *args, **kwargs) -> None:
        pass

    def decay_epsilon(self) -> None:
        pass

    def save(self, path: str | Path) -> None:
        pass

    @classmethod
    def load(cls, path: str | Path) -> "RandomWalkerAgent":
        return cls(n_actions=1)
