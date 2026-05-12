"""
app.py
======
Streamlit dashboard for the Autonomous Bug-Hunter RL project.

Features
--------
* Sidebar: load a trained policy, set episode parameters
  (bug spawn multiplier, max steps, RNG seed), and choose between the
  RL agent and the random-walker baseline for live demos.
* Main canvas: an interactive pyvis graph of the software-module
  network with the agent's current node highlighted, tested nodes
  greyed, and discovered bugs colour-coded by severity.
* Live scoreboard: reward, steps taken, bugs discovered, and a
  step-by-step action log.
* SDG-9 alignment panel.

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from agent import QLearningAgent, RandomWalkerAgent
from env import BugHuntingEnv
from evaluate import SDG_TEXT



st.set_page_config(
    page_title="Bug Hunter RL",
    page_icon="BH",
    layout="wide",
)

ROOT          = Path(__file__).parent.resolve()
DEFAULT_POLICY = ROOT / "models" / "policy_v1.pkl"



NODE_COLORS = {
    "default":    "#3a86ff",   # untested, no known bug
    "tested":     "#a0a0a0",   # tested, no bug
    "minor":      "#ffb703",   # tested, minor bug discovered
    "critical":   "#e63946",   # tested, critical bug discovered
    "agent":      "#2ec4b6",   # node the agent is currently on
    "agent_bug":  "#9b5de5",   # agent on a node where it just found a bug
}


def render_graph(env: BugHuntingEnv, height_px: int = 520) -> str:
    """
    Render the current env state as an interactive pyvis network and
    return its HTML so Streamlit can embed it.

    We rebuild the network on every step rather than mutating in place —
    pyvis is designed for one-shot rendering and trying to mutate an
    existing widget across reruns is more fragile than just regenerating.
    """
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=False,
        notebook=False,
        cdn_resources="in_line",   # avoid external CDN fetches in offline demos
    )
    # Disable physics simulation after initial layout so the graph
    # doesn't keep wobbling between Streamlit reruns.
    net.toggle_physics(True)
    net.barnes_hut(spring_length=120, gravity=-3000)

    found_severity_by_node: Dict[int, str] = {
        node_id: bug.severity for node_id, bug in env.found_bugs
    }

    for node_id in env.graph.nodes():
        name = env.node_names[node_id]
        is_current = (node_id == env.current_node)
        is_tested  = (node_id in env.tested_nodes)
        severity   = found_severity_by_node.get(node_id)

        # Layered colour logic — most-specific state wins.
        if is_current and severity:
            color = NODE_COLORS["agent_bug"]
        elif is_current:
            color = NODE_COLORS["agent"]
        elif severity == "critical":
            color = NODE_COLORS["critical"]
        elif severity == "minor":
            color = NODE_COLORS["minor"]
        elif is_tested:
            color = NODE_COLORS["tested"]
        else:
            color = NODE_COLORS["default"]

        # Visually emphasise the agent and discovered bugs.
        size = 32 if is_current else (26 if severity else 20)

        title = f"<b>{name}</b><br>id: {node_id}"
        if is_tested:
            title += "<br>tested: yes"
        if severity:
            title += f"<br>bug: <b>{severity}</b>"
        if is_current:
            title += "<br><i>agent is here</i>"

        net.add_node(
            node_id,
            label=name,
            color=color,
            size=size,
            title=title,
            borderWidth=3 if is_current else 1,
        )

    for u, v in env.graph.edges():
        net.add_edge(u, v, color="#cccccc")

    # pyvis 0.3.x's write_html() uses the default system encoding,
    # which on Windows (cp1252) can't handle Unicode characters in
    # node tooltips (e.g. ✓).  We use generate_html() to get the
    # HTML string directly — no temp file needed.
    return net.generate_html()



def init_state() -> None:
    """Initialise all session-state slots the app uses."""
    defaults = {
        "env":            None,
        "agent":          None,
        "agent_kind":     None,    # "rl" or "random"
        "running":        False,
        "step_log":       [],      # list of dicts for the action log
        "total_reward":   0.0,
        "episode_done":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_episode(
    n_nodes:    int,
    max_steps:  int,
    bug_mult:   float,
    seed:       int,
    agent_kind: str,
    policy_path: Optional[Path],
) -> None:
    """(Re)build env and agent for a fresh episode."""
    env = BugHuntingEnv(
        n_nodes              = n_nodes,
        max_steps            = max_steps,
        bug_spawn_multiplier = bug_mult,
        seed                 = seed,
    )
    env.reset(seed=seed)

    if agent_kind == "rl":
        if policy_path is None or not Path(policy_path).exists():
            st.sidebar.error(
                f"Policy file not found at: {policy_path}\n\n"
                f"Run `python train.py` first."
            )
            agent = RandomWalkerAgent(n_actions=env.max_degree, seed=seed)
            agent_kind = "random"
        else:
            agent = QLearningAgent.load(policy_path)
    else:
        agent = RandomWalkerAgent(n_actions=env.max_degree, seed=seed)

    st.session_state.env          = env
    st.session_state.agent        = agent
    st.session_state.agent_kind   = agent_kind
    st.session_state.step_log     = []
    st.session_state.total_reward = 0.0
    st.session_state.episode_done = False
    st.session_state.running      = False


def take_one_step() -> None:
    """Advance the simulation by a single env step."""
    env   = st.session_state.env
    agent = st.session_state.agent
    if env is None or agent is None or st.session_state.episode_done:
        return

    state = env._encode_state()
    legal = env.get_legal_actions()
    action = agent.select_action(state, legal, greedy=True)
    out = env.step(action)

    st.session_state.total_reward += out.reward
    st.session_state.step_log.append({
        "step":   env.steps_taken,
        "node":   env.node_names[env.current_node],
        "reward": out.reward,
        "bug":    out.info.get("bug_found"),
    })
    if out.done:
        st.session_state.episode_done = True
        st.session_state.running      = False



def render_sidebar() -> Dict:
    st.sidebar.title("Configuration")

    st.sidebar.subheader("Agent")
    agent_kind_label = st.sidebar.radio(
        "Pick an agent",
        ["RL Agent (Q-Learning)", "Random-Walker Baseline"],
        index=0,
    )
    agent_kind = "rl" if agent_kind_label.startswith("RL") else "random"

    policy_path = st.sidebar.text_input(
        "Policy file (Q-Learning agent only)",
        value=str(DEFAULT_POLICY),
        help="Pickle file produced by train.py",
    )

    st.sidebar.subheader("Environment")
    n_nodes   = st.sidebar.slider("Number of modules (nodes)", 10, 18, 18)
    max_steps = st.sidebar.slider("Max steps per episode",     10, 60, 30)
    bug_mult  = st.sidebar.slider(
        "Bug spawn multiplier", 0.2, 2.0, 1.0, step=0.1,
        help="Scales the per-module bug probability. 1.0 is default.",
    )
    seed = st.sidebar.number_input("Episode seed", value=42, step=1)

    st.sidebar.subheader("Playback")
    step_delay = st.sidebar.slider(
        "Delay between auto-steps (s)", 0.0, 1.5, 0.4, step=0.1
    )

    return {
        "agent_kind":  agent_kind,
        "policy_path": Path(policy_path) if policy_path else None,
        "n_nodes":     n_nodes,
        "max_steps":   max_steps,
        "bug_mult":    bug_mult,
        "seed":        int(seed),
        "step_delay":  step_delay,
    }


def render_legend() -> None:
    """Compact colour legend rendered above the graph."""
    items = [
        ("Agent here",         NODE_COLORS["agent"]),
        ("Agent + bug found",  NODE_COLORS["agent_bug"]),
        ("Critical bug found", NODE_COLORS["critical"]),
        ("Minor bug found",    NODE_COLORS["minor"]),
        ("Tested (no bug)",    NODE_COLORS["tested"]),
        ("Untested",           NODE_COLORS["default"]),
    ]
    chips = "  ".join(
        f"<span style='display:inline-block; width:14px; height:14px; "
        f"background:{c}; border-radius:3px; margin-right:6px; "
        f"vertical-align:middle;'></span>"
        f"<span style='vertical-align:middle; margin-right:18px; "
        f"font-size:0.9em;'>{label}</span>"
        for label, c in items
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_scoreboard() -> None:
    """Three big metric tiles + a progress strip."""
    env = st.session_state.env
    if env is None:
        return

    bugs_found  = len(env.found_bugs)
    bugs_seeded = env.total_bugs_at_start
    critical_n  = sum(1 for _, b in env.found_bugs if b.severity == "critical")
    minor_n     = sum(1 for _, b in env.found_bugs if b.severity == "minor")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reward",      f"{st.session_state.total_reward:.0f}")
    c2.metric("Steps Taken", f"{env.steps_taken} / {env.max_steps}")
    c3.metric("Bugs Found",
              f"{bugs_found} / {bugs_seeded}",
              delta=f"{critical_n} critical, {minor_n} minor",
              delta_color="off")
    progress = bugs_found / bugs_seeded if bugs_seeded else 0.0
    c4.metric("Discovery", f"{progress:.0%}")

    if env.steps_taken > 0:
        st.progress(min(1.0, env.steps_taken / env.max_steps),
                    text="Episode progress")


def render_action_log() -> None:
    log = st.session_state.step_log
    if not log:
        st.info("No steps yet. Click **Run Simulation** or **Step** to begin.")
        return
    rows = []
    for entry in log[-25:]:
        bug = entry["bug"]
        bug_str = "-" if bug is None else bug
        rows.append({
            "step":   entry["step"],
            "node":   entry["node"],
            "reward": f"{entry['reward']:+.0f}",
            "bug":    bug_str,
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


def main() -> None:
    init_state()
    cfg = render_sidebar()

    st.title("Autonomous Bug Hunter - RL Demo")
    st.caption(
        "An RL agent navigates a software-module graph to discover bugs "
        "efficiently. Compare its policy with a random walker, watch it "
        "step through an episode live, and see how it aligns with SDG 9."
    )

    # Build env/agent on first load OR when the user clicks "New Episode".
    if st.session_state.env is None:
        reset_episode(**{k: v for k, v in cfg.items() if k != "step_delay"})

    # ----- Control row ------------------------------------------------- #
    btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1])
    if btn1.button("New Episode", use_container_width=True):
        reset_episode(**{k: v for k, v in cfg.items() if k != "step_delay"})

    if btn2.button("Step", use_container_width=True,
                   disabled=st.session_state.episode_done):
        take_one_step()

    run_clicked = btn3.button(
        "Run Simulation", use_container_width=True,
        disabled=st.session_state.episode_done,
    )
    if run_clicked:
        st.session_state.running = True

    if btn4.button("Stop", use_container_width=True):
        st.session_state.running = False

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Software-Module Graph")
        render_legend()
        graph_html = render_graph(st.session_state.env)
        components.html(graph_html, height=540, scrolling=False)

    with right:
        st.subheader("Live Scoreboard")
        render_scoreboard()

        st.subheader("Action Log")
        render_action_log()

        if st.session_state.episode_done:
            env = st.session_state.env
            st.success(
                f"Episode complete · Reward = {st.session_state.total_reward:.0f} · "
                f"Bugs found = {len(env.found_bugs)} / {env.total_bugs_at_start}"
            )

    with st.expander("SDG 9 - Industry, Innovation and Infrastructure",
                     expanded=False):
        st.text(SDG_TEXT)

    st.caption(
        f"Agent: **{'Q-Learning RL' if st.session_state.agent_kind == 'rl' else 'Random Walker'}**  ·  "
        f"Seed: {cfg['seed']}  ·  Bug-multiplier: {cfg['bug_mult']}"
    )


    if st.session_state.running and not st.session_state.episode_done:
        take_one_step()
        time.sleep(cfg["step_delay"])
        st.rerun()


if __name__ == "__main__":
    main()
