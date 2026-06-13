"""Poll Vertex custom jobs by display-name prefix to terminal state (revision_plan §7).

    python vertex/poll_jobs.py --prefix tactic-<run_id>
Prints each matching job's state; exit 0 only when ALL matched jobs are terminal (SUCCEEDED/
FAILED/CANCELLED). Intended to be called repeatedly by the driving agent with a cooldown between
calls (do not go offline)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactic.common.config import VertexConfig, load_config, load_dotenv  # noqa: E402

TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="display-name prefix, e.g. tactic-<run_id>")
    a = ap.parse_args()
    load_dotenv()
    v = VertexConfig.from_config(load_config())
    cmd = (f'gcloud ai custom-jobs list --region={v.region} --project={v.project_id} '
           f'--filter="displayName:{a.prefix}*" '
           f'--format="json(displayName,state,name)"')
    out = subprocess.check_output(cmd, shell=True, text=True)
    jobs = json.loads(out) if out.strip() else []
    if not jobs:
        print(f"NO JOBS match {a.prefix}*"); sys.exit(2)
    n_term = 0
    for j in jobs:
        st = j.get("state", "?")
        print(f"{st:24s} {j.get('displayName')}")
        if st in TERMINAL:
            n_term += 1
    done = n_term == len(jobs)
    n_ok = sum(1 for j in jobs if j.get("state") == "JOB_STATE_SUCCEEDED")
    print(f"--- {n_term}/{len(jobs)} terminal, {n_ok} succeeded ---")
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
