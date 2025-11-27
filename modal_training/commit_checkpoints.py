#!/usr/bin/env python3
"""
Background script to monitor checkpoint directory and commit Modal volume.
Runs in a separate thread to commit checkpoints as they're saved.
"""

import time
import shutil
from pathlib import Path
from modal import Volume


def monitor_and_commit(
    vol: Volume, checkpoint_dir: str, latest_path: str, check_interval: int = 30
):
    """
    Monitor checkpoint directory and commit volume when new checkpoints appear.

    Args:
        vol: Modal Volume object to commit
        checkpoint_dir: Directory where checkpoints are saved
        latest_path: Path to copy latest checkpoint to
        check_interval: How often to check for new checkpoints (seconds)
    """
    checkpoint_path = Path(checkpoint_dir)
    latest_checkpoint_path = Path(latest_path)
    latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    last_commit_time = 0
    last_seen_files = set()

    print("Starting checkpoint monitor: watching", checkpoint_path)

    while True:
        try:
            if checkpoint_path.exists():
                # Find all checkpoint files
                checkpoint_files = list(checkpoint_path.glob("*.ckpt"))

                if checkpoint_files:
                    # Find the latest checkpoint by modification time
                    latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                    latest_mtime = latest.stat().st_mtime

                    # Check if this is a new checkpoint or updated one
                    is_new = latest.name not in last_seen_files
                    is_updated = latest_mtime > last_commit_time

                    if is_new or is_updated:
                        # Copy to latest-checkpoint location
                        shutil.copy2(latest, latest_checkpoint_path)
                        print(
                            f"✓ Copied latest checkpoint: {latest.name} -> latest-checkpoint.ckpt"
                        )

                        # Commit the volume
                        vol.commit()
                        print("✓ Committed checkpoints to Modal volume")

                        last_commit_time = latest_mtime
                        last_seen_files.add(latest.name)

        except Exception as e:
            print(f"Warning: Error in checkpoint monitor: {e}")

        time.sleep(check_interval)
