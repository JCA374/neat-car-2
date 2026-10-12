"""
Train steps one after another (each with scripts/run_step.py).

    python scripts/run_queue.py                         # every step that has no result.json yet, in id order
    python scripts/run_queue.py n01_lap_time p01_lap_time
    nohup python scripts/run_queue.py > outputs/queue.log 2>&1 &   # keeps running after the terminal closes

Writes outputs/queue.json (what's queued, running, done) for scripts/progress.py.
A failed step is recorded and the queue moves on to the next one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carracing.steps import all_steps, fmt_duration  # noqa: E402

STATUS = "outputs/queue.json"


def save(status: dict) -> None:
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(status, fh, indent=1)
    os.replace(tmp, STATUS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", help="Step ids (default: all steps without a result yet).")
    args = parser.parse_args()

    steps = {s.id: s for s in all_steps()}
    if args.steps:
        missing = [i for i in args.steps if i not in steps]
        if missing:
            raise SystemExit(f"Unknown step(s): {', '.join(missing)}")
        queue = [steps[i] for i in args.steps]
    else:
        queue = [s for s in sorted(steps.values(), key=lambda s: s.id)
                 if not os.path.exists(os.path.join(s.out_dir, "result.json")) and not s.imported]
    if not queue:
        print("Nothing to train: every step has a result.")
        return 0

    status = {"pid": os.getpid(), "started": time.time(), "finished": None,
              "steps": [{"id": s.id, "model": s.model, "title": s.title, "state": "queued",
                         "started": None, "ended": None} for s in queue]}
    save(status)
    print(f"Queue: {', '.join(s.id for s in queue)}", flush=True)
    for step, entry in zip(queue, status["steps"]):
        entry.update(state="running", started=time.time())
        save(status)
        print(f"\n=== {step.id}: {step.title} ===", flush=True)
        code = subprocess.call([sys.executable, os.path.join(os.path.dirname(__file__), "run_step.py"), step.path])
        entry.update(state="done" if code == 0 else f"failed (exit {code})", ended=time.time())
        save(status)
        print(f"=== {step.id}: {entry['state']} after {fmt_duration(entry['ended'] - entry['started'])} ===", flush=True)
    status["finished"] = time.time()
    save(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
