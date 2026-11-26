"""Stop detached fast evaluator groups that still hold this run's evidence logs."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def recorded_groups(folder):
    startup = folder / "startup.jsonl"
    groups = set()
    if startup.exists():
        for line in startup.read_text().splitlines():
            try:
                row = json.loads(line)
                groups.update((int(row["worker_pid"]), int(row["server_pid"])))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # The parent may have been interrupted in the middle of a write.
    return groups


def log_writers(logs):
    if not logs:
        return
    result = subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-F", "pfa", "-a", "-d", "1,2", "--", *logs],
        capture_output=True, text=True, check=False,
    )
    if result.returncode and result.stderr:
        raise RuntimeError(result.stderr.strip())
    pid = None
    for field in result.stdout.splitlines():
        if field.startswith("p"):
            pid = int(field[1:])
        elif field == "aw" and pid is not None:
            yield pid


def stop_groups(evidence):
    for folder in sorted(evidence.iterdir()):
        # Matchup creates one timestamp/PID/UUID folder per attempt. A retry can
        # leave evidence from more than one attempt under this invocation's root.
        if not folder.is_dir() or folder.is_symlink():
            continue
        recorded = recorded_groups(folder)
        logs = [*folder.glob("worker-*.log"), *folder.glob("server-*.log")]
        for pid in set(log_writers(logs)):
            try:
                group = os.getpgid(pid)
                # Checking the open log guards against stale PIDs in startup.jsonl.
                # A group leader can own a log before its startup record is written.
                if group > 1 and group != os.getpgrp() and (group in recorded or group == pid):
                    os.killpg(group, signal.SIGKILL)
                    print(f"Stopped fast evaluator process group {group} ({folder.name})", flush=True)
            except ProcessLookupError:
                continue


if __name__ == "__main__":
    stop_groups(Path(sys.argv[1]))
