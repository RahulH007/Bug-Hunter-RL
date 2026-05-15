"""
rollback.py
===========
Policy snapshot manager — save, list, restore, and delete versioned
policy snapshots so you can roll back to any previous training run.

Snapshots are stored under models/snapshots/ as pairs:
    <timestamp>_<tag>.pkl    — the Q-table pickle
    <timestamp>_<tag>.json   — metadata (run_id, metrics, tag, created_at)

Commands
--------
    python rollback.py list                      list all snapshots
    python rollback.py save [--tag TAG]          snapshot current policy
    python rollback.py restore <snapshot_id>     restore snapshot → policy_v1.pkl
    python rollback.py delete  <snapshot_id>     delete a snapshot
    python rollback.py info    <snapshot_id>     print snapshot metadata

The snapshot_id is the filename stem shown by `list`
(e.g. 20260508_123456_baseline or 20260508_123456_<run-uuid-prefix>).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


ROOT          = Path(__file__).parent.resolve()
POLICY_PATH   = ROOT / "models" / "policy_v1.pkl"
SNAPSHOT_DIR  = ROOT / "models" / "snapshots"
LOG_PATH      = ROOT / "logs" / "results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _latest_run_record() -> Optional[Dict]:
    """Return the most recent entry from the MLOps training log, or None."""
    if not LOG_PATH.exists():
        return None
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            history = json.load(f)
        return history[-1] if isinstance(history, list) and history else None
    except (json.JSONDecodeError, OSError):
        return None


def _list_snapshots() -> List[Path]:
    """Sorted list of snapshot metadata files (newest first)."""
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("*.json"), reverse=True)


def _snapshot_id_to_paths(snap_id: str):
    """Return (pkl_path, json_path) for a given snapshot id stem."""
    pkl  = SNAPSHOT_DIR / f"{snap_id}.pkl"
    meta = SNAPSHOT_DIR / f"{snap_id}.json"
    return pkl, meta


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(_args: argparse.Namespace) -> int:
    snapshots = _list_snapshots()
    if not snapshots:
        print("No snapshots found. Run `python rollback.py save` to create one.")
        return 0

    print(f"{'ID (stem)':<40} {'Tag':<20} {'Episodes':>8} {'Discovery':>10} {'Created'}")
    print("-" * 100)
    for meta_path in snapshots:
        try:
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            episodes  = m.get("hyperparameters", {}).get("episodes", "?")
            disc_rate = m.get("metrics", {}).get("bug_discovery_rate", None)
            disc_str  = f"{disc_rate:.2%}" if disc_rate is not None else "?"
            tag       = m.get("tag", "")
            created   = m.get("created_at", "?")[:19]
            print(f"{meta_path.stem:<40} {tag:<20} {str(episodes):>8} {disc_str:>10} {created}")
        except (json.JSONDecodeError, OSError):
            print(f"{meta_path.stem:<40} [corrupted metadata]")
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    if not POLICY_PATH.exists():
        print(f"ERROR: No policy found at {POLICY_PATH}. Run `python train.py` first.")
        return 1

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.tag or ""

    # Fall back to run-id prefix if no tag provided
    run = _latest_run_record()
    if not tag and run:
        tag = run.get("run_id", "")[:8]
    if not tag:
        tag = "manual"

    stem     = f"{ts}_{tag}"
    pkl_dst  = SNAPSHOT_DIR / f"{stem}.pkl"
    meta_dst = SNAPSHOT_DIR / f"{stem}.json"

    shutil.copy2(POLICY_PATH, pkl_dst)

    metadata: Dict = {
        "snapshot_id": stem,
        "tag":         tag,
        "created_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source":      str(POLICY_PATH),
    }
    if run:
        metadata["run_id"]         = run.get("run_id")
        metadata["hyperparameters"] = run.get("hyperparameters", {})
        metadata["metrics"]         = run.get("metrics", {})

    with open(meta_dst, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Snapshot saved: {stem}")
    print(f"  Policy  → {pkl_dst}")
    print(f"  Metadata→ {meta_dst}")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    pkl_src, meta_src = _snapshot_id_to_paths(args.snapshot_id)
    if not pkl_src.exists():
        print(f"ERROR: Snapshot not found: {pkl_src}")
        print("Run `python rollback.py list` to see available snapshots.")
        return 1

    # Auto-backup the current policy before overwriting
    if POLICY_PATH.exists():
        backup = cmd_save(argparse.Namespace(tag="pre-restore"))
        if backup != 0:
            print("WARNING: Could not auto-backup current policy. Proceeding anyway.")

    shutil.copy2(pkl_src, POLICY_PATH)
    print(f"Restored snapshot '{args.snapshot_id}' → {POLICY_PATH}")

    if meta_src.exists():
        with open(meta_src, encoding="utf-8") as f:
            m = json.load(f)
        episodes  = m.get("hyperparameters", {}).get("episodes", "?")
        disc_rate = m.get("metrics", {}).get("bug_discovery_rate", None)
        print(f"  Tag: {m.get('tag', '')} | Episodes: {episodes} "
              f"| Discovery rate: {f'{disc_rate:.2%}' if disc_rate else '?'}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    pkl, meta = _snapshot_id_to_paths(args.snapshot_id)
    deleted_any = False
    for p in (pkl, meta):
        if p.exists():
            p.unlink()
            print(f"Deleted: {p}")
            deleted_any = True
    if not deleted_any:
        print(f"ERROR: Snapshot '{args.snapshot_id}' not found.")
        return 1
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    _, meta = _snapshot_id_to_paths(args.snapshot_id)
    if not meta.exists():
        print(f"ERROR: Metadata not found for snapshot '{args.snapshot_id}'.")
        return 1
    with open(meta, encoding="utf-8") as f:
        m = json.load(f)
    print(json.dumps(m, indent=2))
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Policy snapshot manager for Bug-Hunter RL."
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all saved snapshots.")

    save_p = sub.add_parser("save", help="Snapshot the current policy.")
    save_p.add_argument("--tag", default="", help="Human-readable label for this snapshot.")

    restore_p = sub.add_parser("restore", help="Restore a snapshot to models/policy_v1.pkl.")
    restore_p.add_argument("snapshot_id", help="Stem from `rollback.py list`.")

    delete_p = sub.add_parser("delete", help="Delete a snapshot.")
    delete_p.add_argument("snapshot_id")

    info_p = sub.add_parser("info", help="Print snapshot metadata.")
    info_p.add_argument("snapshot_id")

    return p.parse_args()


def main() -> None:
    args    = parse_args()
    command = {
        "list":    cmd_list,
        "save":    cmd_save,
        "restore": cmd_restore,
        "delete":  cmd_delete,
        "info":    cmd_info,
    }[args.command]
    sys.exit(command(args))


if __name__ == "__main__":
    main()
