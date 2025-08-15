#!/usr/bin/env python3
"""
Simple autoscaler for FORCEPS workers + index builder.

Logic:
- Reads app/config.yaml for Redis queues and limits
- Periodically checks Redis LLEN for job/results queues
- Scales worker processes up/down based on job backlog
- Starts index builder when results backlog appears and builder not running

Only manages processes it spawns (tracked in ./cache/autoscaler_pids.json).
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
import redis

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
PID_FILE = CACHE / "autoscaler_pids.json"


def load_pids():
    if PID_FILE.exists():
        try:
            return json.loads(PID_FILE.read_text())
        except Exception:
            return {"workers": [], "builder": None}
    return {"workers": [], "builder": None}


def save_pids(pids):
    PID_FILE.write_text(json.dumps(pids))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def start_worker(config_path: Path) -> int:
    log_path = CACHE / f"engine_{int(time.time())}.log"
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "app" / "engine.py"), "--config", str(config_path)],
            stdout=log,
            stderr=log,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
    return proc.pid


def kill_pid(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def start_builder(config_path: Path) -> int:
    log_path = CACHE / f"builder_{int(time.time())}.log"
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "app" / "build_index.py"), "--config", str(config_path)],
            stdout=log,
            stderr=log,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
    return proc.pid


def main():
    config_path = ROOT / "app" / "config.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = yaml.safe_load(open(config_path, "r"))
    cfg_redis = config["redis"]

    r = redis.Redis(host=cfg_redis["host"], port=cfg_redis["port"], db=0)
    job_q = cfg_redis["job_queue"]
    res_q = cfg_redis["results_queue"]

    # Limits
    cpu_count = max(1, (os.cpu_count() or 2) - 0)
    min_workers = 1
    max_workers = min(4, cpu_count)

    pids = load_pids()

    while True:
        try:
            jobs = int(r.llen(job_q))
            results = int(r.llen(res_q))
        except Exception as e:
            print(f"Redis error: {e}")
            time.sleep(5)
            continue

        # Reap dead workers
        pids["workers"] = [pid for pid in pids.get("workers", []) if pid_alive(pid)]

        # Target workers: 1 per 2 queued jobs, bounded
        target = max(min_workers, min(max_workers, (jobs + 1) // 2))

        # Scale up
        while len(pids["workers"]) < target:
            pid = start_worker(config_path)
            pids["workers"].append(pid)
            save_pids(pids)

        # Scale down (leave at least min_workers)
        while len(pids["workers"]) > max(min_workers, target):
            pid = pids["workers"].pop()
            kill_pid(pid)
            save_pids(pids)

        # Start builder when there are results and no builder running
        if results > 0:
            builder_pid = pids.get("builder")
            if not (builder_pid and pid_alive(builder_pid)):
                pid = start_builder(config_path)
                pids["builder"] = pid
                save_pids(pids)

        time.sleep(10)


if __name__ == "__main__":
    main()


