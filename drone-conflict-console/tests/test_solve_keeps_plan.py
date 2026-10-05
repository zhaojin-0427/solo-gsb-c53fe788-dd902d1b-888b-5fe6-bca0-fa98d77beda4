"""Route-level guard: a timed-out solve must not overwrite a known plan.

Uses a temporary SQLite store and monkeypatches ``app.solver.run_solve`` so
that a first (real, tiny) solve is followed by a re-solve whose probe timed
out (``status == "unknown"``).  The stored plan must remain the previously
proven feasible table, and it must still be usable for batch creation.

Run:  python tests/test_solve_keeps_plan.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import main as web  # noqa: E402
from app.solver import SolveResult  # noqa: E402

FAILS = 0


def check(cond, msg):
    global FAILS
    if not cond:
        FAILS += 1
        print("FAIL:", msg)
    else:
        print("ok:", msg)


CROSSROADS = {
    "safety_radius": 30,
    "safety_height": 10,
    "sorties": [
        {"id": "A", "weight": 1, "window_hi": 20,
         "waypoints": [
             {"x": 0, "y": 0, "z": 20, "t": 0},
             {"x": 200, "y": 0, "z": 20, "t": 20}]},
        {"id": "B", "weight": 1, "window_hi": 20,
         "waypoints": [
             {"x": 100, "y": -100, "z": 20, "t": 0},
             {"x": 100, "y": 100, "z": 20, "t": 20}]},
    ],
}


def main():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "console.db")
    web.DB_PATH = db_path
    web.store = web.Store(db_path)

    with TestClient(web.app) as client:
        r = client.post("/api/import", json=CROSSROADS)
        check(r.status_code == 200, f"import ok ({r.status_code})")
        sid = r.json()["scenario_id"]

        # First solve: real solver, known feasible plan persisted.
        r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20})
        check(r.status_code == 200, f"first solve ok ({r.status_code})")
        first = r.json()["solution"]
        check(first["feasible"], "first solve is feasible")
        saved_offsets = dict(first["offsets"])

        # Re-solve whose probe times out: nothing proven, nothing known.
        unknown = SolveResult(feasible=False, status="unknown")
        original = web.run_solve

        def fake_unknown(*args, **kwargs):
            return unknown

        web.run_solve = fake_unknown
        try:
            r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20})
        finally:
            web.run_solve = original
        check(r.status_code == 200, f"unknown solve still 200 ({r.status_code})")
        # The response carries the *retained* feasible plan so the page keeps
        # rendering a usable schedule immediately.
        body = r.json()["solution"]
        check(body["status"] in ("optimal", "feasible") and body["feasible"],
              "response falls back to the retained feasible plan on unknown")

        # SQLite still holds the old plan.
        r = client.get(f"/api/scenarios/{sid}")
        persisted = r.json()["solution"]
        check(persisted["feasible"] and persisted["offsets"] == saved_offsets,
              "unknown re-solve never overwrote the stored plan")

        # Batches keep being created from the retained plan.
        r = client.post(f"/api/scenarios/{sid}/batches", json={"name": "kept"})
        check(r.status_code == 201, f"batch created from retained plan ({r.status_code})")
        check(r.json()["planned_offsets"] == saved_offsets,
              "batch frozen on the retained (non-overwritten) plan")

        # A genuine infeasibility proof still replaces the stored result.
        infeas = SolveResult(feasible=False, status="infeasible",
                             cores=[{"sorties": ["A", "B"], "size": 2,
                                     "mutex_offsets": []}],
                             truncated=False)
        web.run_solve = lambda *a, **k: infeas  # noqa: E731
        r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20})
        web.run_solve = original
        check(r.status_code == 200 and not r.json()["solution"]["feasible"],
              "proven infeasible replaces the stored result (distinct from unknown)")
        r = client.get(f"/api/scenarios/{sid}")
        check(r.json()["solution"]["status"] == "infeasible",
              "infeasible proof persisted, so 'no solution' stays visible")

    print(f"route guard checks, failures: {FAILS}")
    return FAILS


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
