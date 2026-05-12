

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Module catalogue: name + bug-spawn weight (higher = more likely to be buggy)
# ---------------------------------------------------------------------------
MODULE_CATALOGUE: List[Tuple[str, float]] = [
    ("Auth",            0.35),
    ("Database",        0.85),  # high-risk
    ("API_Gateway",     0.80),  # high-risk
    ("UI_Frontend",     0.25),
    ("Payment",         0.75),  # high-risk
    ("Logging",         0.15),
    ("Cache",           0.40),
    ("Search",          0.45),
    ("Notifications",   0.30),
    ("UserService",     0.50),
    ("OrderService",    0.65),
    ("Inventory",       0.55),
    ("Analytics",       0.30),
    ("Recommendation",  0.40),
    ("FileStorage",     0.45),
    ("Messaging",       0.35),
    ("AdminPanel",      0.25),
    ("ReportingEngine", 0.50),
]


@dataclass
class BugInfo:
    """Lightweight container for a single bug instance on a node."""
    severity: str         # "critical" or "minor"
    reward:   int         # +50 critical, +10 minor


@dataclass
class StepResult:
    """Structured return from env.step() to keep call-sites readable."""
    state:   Tuple[int, int, int, int]
    reward:  float
    done:    bool
    info:    Dict = field(default_factory=dict)


class BugHuntingEnv:
    """
    Graph-based bug hunting environment.

    State (returned to the agent)
    -----------------------------
    A discretised 4-tuple suitable for Q-table indexing:
        (current_node_id, steps_remaining_bucket,
         tested_count_bucket, current_node_already_tested)

    The fourth component is critical: without it, the agent cannot
    distinguish "I'm at Database and it's untested" from "I'm at Database
    and I already tested it" — and would happily oscillate between two
    high-value nodes forever. With this flag the Q-table separates those
    two situations and the policy learns to move on after testing.

    Action space
    ------------
    Discrete integer in [0, max_degree). Action `a` means "move to and test
    the a-th neighbour of the current node (in sorted order)". If the
    chosen neighbour index is out of bounds (current node has fewer
    neighbours than the agent's max action dim), the action is clipped to a
    valid neighbour to keep the episode well-defined.

    Rewards
    -------
    * +50  for discovering a critical bug
    * +10  for discovering a minor bug
    * -1   per step (encourages short paths)
    * +0   for re-visiting an already-tested node (still pays the -1 step cost)
    """

    # Reward constants — exposed at class level so train/evaluate can import.
    REWARD_CRITICAL = 50
    REWARD_MINOR    = 10
    STEP_PENALTY    = -1

    # Discretisation buckets for the steps-remaining and tested-count
    # components of the state tuple. Keeping these small keeps the Q-table
    # tractable while still giving the agent useful temporal signal.
    N_TIME_BUCKETS   = 4
    N_TESTED_BUCKETS = 4

    def __init__(
        self,
        n_nodes: int = 18,
        max_steps: int = 30,
        bug_spawn_multiplier: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_nodes : int
            Number of software-module nodes in the graph (15–20 recommended).
        max_steps : int
            Episode horizon — number of test/move actions the agent gets.
        bug_spawn_multiplier : float
            Global scalar applied to per-module bug probabilities. Useful
            for the Streamlit UI slider.
        seed : Optional[int]
            Master seed for reproducibility. The graph topology and the
            bug-spawn process are both seeded from this.
        """
        if not 5 <= n_nodes <= len(MODULE_CATALOGUE):
            raise ValueError(
                f"n_nodes must be in [5, {len(MODULE_CATALOGUE)}], got {n_nodes}"
            )

        self.n_nodes              = n_nodes
        self.max_steps            = max_steps
        self.bug_spawn_multiplier = bug_spawn_multiplier
        self._master_seed         = seed
        self._rng                 = random.Random(seed)
        self._np_rng              = np.random.default_rng(seed)

        # Build the graph topology ONCE — its structure is fixed across
        # episodes (only bug placements re-randomise on reset).
        self.graph: nx.Graph              = self._build_graph()
        self.node_names: Dict[int, str]   = {
            i: MODULE_CATALOGUE[i][0] for i in range(n_nodes)
        }
        self.bug_weights: Dict[int, float] = {
            i: MODULE_CATALOGUE[i][1] for i in range(n_nodes)
        }
        self.max_degree = max(dict(self.graph.degree()).values())

        # Per-episode mutable state
        self.bugs: Dict[int, BugInfo]   = {}
        self.tested_nodes: Set[int]     = set()
        self.found_bugs: List[Tuple[int, BugInfo]] = []
        self.current_node: int          = 0
        self.steps_taken: int           = 0
        self.total_bugs_at_start: int   = 0

    def _build_graph(self) -> nx.Graph:
        """
        Build a connected graph that loosely resembles a microservice
        architecture: a backbone of well-connected core services
        (API_Gateway, Database, Auth) plus peripheral modules.
        """
        # We use a fixed seed for the topology so the *structure* is
        # reproducible across runs even if bug placements vary.
        topology_seed = (
            self._master_seed if self._master_seed is not None else 42
        )
        g = nx.connected_watts_strogatz_graph(
            n=self.n_nodes,
            k=4,                  # each node connected to 4 nearest neighbours
            p=0.3,                # rewiring probability — adds long-range edges
            tries=100,
            seed=topology_seed,
        )

        # Add a few extra edges into "hub" nodes (Database=1, API_Gateway=2)
        # to mimic the high fan-in of core infrastructure services.
        hub_indices = [i for i in (1, 2) if i < self.n_nodes]
        rng = random.Random(topology_seed)
        for hub in hub_indices:
            for other in rng.sample(range(self.n_nodes), k=min(3, self.n_nodes - 1)):
                if other != hub and not g.has_edge(hub, other):
                    g.add_edge(hub, other)

        assert nx.is_connected(g), "Generated graph must be connected."
        return g

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                  #
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Tuple[int, int, int, int]:
        """
        Start a new episode. Re-rolls bug placements but keeps graph
        topology fixed.

        Parameters
        ----------
        seed : Optional[int]
            If provided, re-seeds the per-episode RNG. Useful for the
            evaluation script which runs both agents on identical episodes.
        """
        if seed is not None:
            self._rng    = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        self.bugs        = self._spawn_bugs()
        self.tested_nodes = set()
        self.found_bugs   = []
        self.steps_taken  = 0
        # Always start at node 0 (conventionally "Auth" — the entry point).
        self.current_node       = 0
        self.total_bugs_at_start = len(self.bugs)
        return self._encode_state()

    def _spawn_bugs(self) -> Dict[int, BugInfo]:
        """Sample a bug for each node based on its risk weight."""
        bugs: Dict[int, BugInfo] = {}
        for node_id, weight in self.bug_weights.items():
            p = min(1.0, weight * self.bug_spawn_multiplier)
            if self._rng.random() < p:
                # 30% of bugs on a given node are critical, rest are minor.
                if self._rng.random() < 0.30:
                    bugs[node_id] = BugInfo("critical", self.REWARD_CRITICAL)
                else:
                    bugs[node_id] = BugInfo("minor", self.REWARD_MINOR)
        return bugs

    def step(self, action: int) -> StepResult:
        """
        Execute one action.

        Parameters
        ----------
        action : int
            Index of the neighbour to move to (sorted neighbour list).
            Clipped to a legal neighbour if out of range.
        """
        neighbours = self.get_legal_actions()
        # Defensive clipping: if the agent picks an out-of-range action,
        # we map it onto the available neighbour set rather than crashing.
        if not neighbours:
            # Isolated node (should not happen — graph is connected).
            return StepResult(self._encode_state(), 0.0, True, {"reason": "isolated"})
        action = action % len(neighbours)
        next_node = neighbours[action]

        self.current_node = next_node
        self.steps_taken += 1
        reward = float(self.STEP_PENALTY)

        info: Dict = {"node": self.node_names[next_node], "bug_found": None}

        # Test the node we just moved into (only first visit pays out).
        if next_node not in self.tested_nodes:
            self.tested_nodes.add(next_node)
            if next_node in self.bugs:
                bug = self.bugs[next_node]
                reward += bug.reward
                self.found_bugs.append((next_node, bug))
                info["bug_found"] = bug.severity

        done = (
            self.steps_taken >= self.max_steps
            or len(self.found_bugs) == self.total_bugs_at_start
        )
        return StepResult(self._encode_state(), reward, done, info)


    def _encode_state(self) -> Tuple[int, int, int, int]:
        """
        Discretise the raw state into a Q-table-friendly tuple:
            (current_node_id, time_bucket, tested_bucket, current_tested_flag)

        The fourth component (0/1) tells the agent whether the node it is
        currently *standing on* has already been tested — without this,
        two situations that demand opposite actions ("test this node" vs
        "leave, you already tested it") would share the same Q-row and
        the policy would oscillate.
        """
        steps_left = max(0, self.max_steps - self.steps_taken)
        time_bucket = min(
            self.N_TIME_BUCKETS - 1,
            int(self.N_TIME_BUCKETS * steps_left / max(1, self.max_steps)),
        )
        tested_bucket = min(
            self.N_TESTED_BUCKETS - 1,
            int(self.N_TESTED_BUCKETS * len(self.tested_nodes) / max(1, self.n_nodes)),
        )
        current_tested = 1 if self.current_node in self.tested_nodes else 0
        return (self.current_node, time_bucket, tested_bucket, current_tested)

    def get_legal_actions(self) -> List[int]:
        """Sorted list of neighbour node-ids for the current node."""
        return sorted(self.graph.neighbors(self.current_node))

    def total_possible_reward(self) -> int:
        """Maximum reward an oracle could obtain on the current episode."""
        return sum(b.reward for b in self.bugs.values())

    def snapshot(self) -> Dict:
        """Serializable snapshot of the env state — useful for the UI."""
        return {
            "current_node":   self.current_node,
            "current_name":   self.node_names[self.current_node],
            "steps_taken":    self.steps_taken,
            "max_steps":      self.max_steps,
            "tested_nodes":   sorted(self.tested_nodes),
            "found_bugs":     [
                {"node": n, "name": self.node_names[n], "severity": b.severity}
                for n, b in self.found_bugs
            ],
            "remaining_bugs": self.total_bugs_at_start - len(self.found_bugs),
        }


if __name__ == "__main__":
    # Quick sanity check
    env = BugHuntingEnv(n_nodes=18, max_steps=30, seed=7)
    s = env.reset()
    print(f"Initial state: {s}")
    print(f"Bugs seeded:   {len(env.bugs)} "
          f"(critical={sum(1 for b in env.bugs.values() if b.severity == 'critical')})")
    print(f"Max degree:    {env.max_degree}")
    total = 0.0
    for _ in range(env.max_steps):
        legal = env.get_legal_actions()
        a = random.randrange(len(legal))
        out = env.step(a)
        total += out.reward
        if out.done:
            break
    print(f"Random rollout total reward: {total:.1f}")
    print(f"Bugs found: {len(env.found_bugs)} / {env.total_bugs_at_start}")
