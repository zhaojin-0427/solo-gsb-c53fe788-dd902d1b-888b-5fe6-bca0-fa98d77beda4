"""Staged solve-result adjudication and timeout persistence.

Covers:

* every solve stage keeps the best witnessed take-off table consistent with
  the constraints proven so far; a new witness replaces it only when its
  weighted delay is no worse (scripted solver stub, deterministic);
* a timed-out optimisation stage falls back to the previously witnessed
  table instead of reporting ``unknown``;
* a lexicographic round that only reaches FEASIBLE adopts the witness but
  neither locks the unproven offset nor marks the result ``optimal``; a
  round that reaches UNKNOWN keeps the previous candidate;
* ``optimal`` is reported only when the weighted objective AND every
  lexicographic round are proven OPTIMAL; ``infeasible`` and ``unknown``
  stay distinct;
* through the HTTP API: an inconclusive re-solve (``unknown``) never
  overwrites a result already stored in SQLite, while conclusive results
  (optimal / feasible / infeasible) are persisted and gate batch creation.

Run:  python tests/test_solve_stages.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ortools.sat.python import cp_model  # noqa: E402

from app.model import Waypoint, make_sortie  # noqa: E402
from app.solver import (  # noqa: E402
    SolveResult,
    build_pair_constraints,
    run_solve,
    scan_schedule,
)
import app.solver as solver_mod  # noqa: E402

FAILS = 0


def check(cond, msg):
    global FAILS
    if not cond:
        FAILS += 1
        print("FAIL:", msg)


# ---------------------------------------------------------------------------
# Scripted solver stub
# ---------------------------------------------------------------------------


class StubSolver:
    """Stand-in for ``cp_model.CpSolver``; one scripted ``Solve`` per instance."""

    def __init__(self, status, values=None):
        self.parameters = types.SimpleNamespace(
            max_time_in_seconds=0.0, num_search_workers=0)
        self._status = status
        self._values = values or {}

    def Solve(self, model):  # noqa: N802 (mirrors CpSolver)
        return self._status

    def Value(self, var):  # noqa: N802
        return self._values[var.Name()]

    def StatusName(self, status):  # noqa: N802
        return {cp_model.UNKNOWN: "UNKNOWN"}.get(status, str(status))


def run_with_script(sorties, pcs, script, **kwargs):
    """Run ``run_solve`` with ``_new_solver`` returning scripted stubs.

    ``script`` is a list of ``(status, {var_name: value})`` consumed in
    order, one per solver instance (probe, weighted objective, lex rounds).
    Returns ``(result, leftovers)`` so tests can assert the solve stopped
    consuming stages once the proof became impossible.
    """
    queue = list(script)
    original = solver_mod._new_solver

    def factory(time_limit):
        if not queue:
            raise AssertionError("run_solve requested more solver stages than scripted")
        status, values = queue.pop(0)
        return StubSolver(status, values)

    solver_mod._new_solver = factory
    try:
        res = run_solve(sorties, pcs, **kwargs)
    finally:
        solver_mod._new_solver = original
    return res, queue


def two_sorties():
    """A and B, both weight 1, window [0, 10]; geometry never conflicts."""
    a = make_sortie("A", 1, 10, [Waypoint(0, 0, 20, 0), Waypoint(100, 0, 20, 10)])
    b = make_sortie("B", 1, 10, [Waypoint(0, 500, 20, 0), Waypoint(100, 500, 20, 10)])
    return [a, b]


TO = {"A": "to_A", "B": "to_B"}  # variable names used by _build_model


def vals(a, b):
    return {TO["A"]: a, TO["B"]: b}


def test_stage1_unknown_falls_back_to_probe_witness():
    sorties = two_sorties()
    res, left = run_with_script(sorties, [], [
        (cp_model.FEASIBLE, vals(2, 3)),   # probe witnesses a table
        (cp_model.UNKNOWN, None),          # weighted objective times out
    ], time_limit=30.0)
    check(res.feasible and res.status == "feasible",
          f"stage-1 timeout must fall back to probe witness, got {res.status}")
    check(res.offsets == {"A": 2, "B": 3},
          f"fallback must keep the probe table, got {res.offsets}")
    check(res.weighted_delay == 5, f"delay of probe table, got {res.weighted_delay}")
    check(left == [], f"no lex stage without proven objective, left={left}")


def test_stage1_feasible_witness_replaces_only_when_not_worse():
    sorties = two_sorties()
    # Better witness (delay 2 < 5) replaces the probe table.
    res, _ = run_with_script(sorties, [], [
        (cp_model.FEASIBLE, vals(2, 3)),
        (cp_model.FEASIBLE, vals(1, 1)),
    ], time_limit=30.0)
    check(res.status == "feasible" and res.offsets == {"A": 1, "B": 1},
          f"better stage-1 witness must be adopted, got {res.status} {res.offsets}")
    # Worse witness (delay 8 > 2) must NOT replace the probe table.
    res, _ = run_with_script(sorties, [], [
        (cp_model.FEASIBLE, vals(1, 1)),
        (cp_model.FEASIBLE, vals(4, 4)),
    ], time_limit=30.0)
    check(res.offsets == {"A": 1, "B": 1} and res.weighted_delay == 2,
          f"worse stage-1 witness must not replace, got {res.offsets}")


def test_all_stages_optimal_marks_optimal():
    sorties = two_sorties()
    res, left = run_with_script(sorties, [], [
        (cp_model.FEASIBLE, vals(2, 3)),
        (cp_model.OPTIMAL, vals(1, 2)),    # proven optimum, delay 3
        (cp_model.OPTIMAL, vals(0, 3)),    # lex round A proven, locked
        (cp_model.OPTIMAL, vals(0, 3)),    # lex round B proven, locked
    ], time_limit=30.0)
    check(res.status == "optimal", f"fully proven run must be optimal, got {res.status}")
    check(res.offsets == {"A": 0, "B": 3} and res.weighted_delay == 3,
          f"lex witness expected, got {res.offsets} d={res.weighted_delay}")
    check(left == [], f"all stages consumed, left={left}")


def test_lex_feasible_round_not_locked_not_optimal():
    sorties = two_sorties()
    res, left = run_with_script(sorties, [], [
        (cp_model.FEASIBLE, vals(2, 3)),
        (cp_model.OPTIMAL, vals(1, 2)),
        (cp_model.FEASIBLE, vals(0, 3)),   # lex round A: witness, unproven
        (cp_model.OPTIMAL, vals(0, 3)),    # must remain unconsumed: loop broke
    ], time_limit=30.0)
    check(res.feasible and res.status == "feasible",
          f"unproven lex round must not be marked optimal, got {res.status}")
    check(res.offsets == {"A": 0, "B": 3},
          f"valid lex witness should be adopted, got {res.offsets}")
    check(len(left) == 1, "after an unproven lex round no further round may run")


def test_lex_unknown_round_keeps_previous_candidate():
    sorties = two_sorties()
    res, left = run_with_script(sorties, [], [
        (cp_model.FEASIBLE, vals(2, 3)),
        (cp_model.OPTIMAL, vals(1, 2)),
        (cp_model.OPTIMAL, vals(0, 3)),    # lex round A proven
        (cp_model.UNKNOWN, None),          # lex round B times out
    ], time_limit=30.0)
    check(res.status == "feasible",
          f"lex timeout must downgrade to feasible, got {res.status}")
    check(res.offsets == {"A": 0, "B": 3},
          f"candidate must come from the last proven round, got {res.offsets}")
    check(res.weighted_delay == 3, f"delay, got {res.weighted_delay}")


def test_probe_unknown_stays_unknown():
    sorties = two_sorties()
    res, _ = run_with_script(sorties, [], [
        (cp_model.UNKNOWN, None),
    ], time_limit=30.0)
    check(not res.feasible and res.status == "unknown",
          f"empty probe must stay unknown, got {res.status}")
    check(res.offsets == {} and res.weighted_delay is None,
          f"unknown carries no table, got {res.offsets}")


# ---------------------------------------------------------------------------
# Real-solver behaviour
# ---------------------------------------------------------------------------


def crossing_sorties():
    """A eastbound, B northbound, crossing at (100, 0) at the same altitude."""
    a = make_sortie("A", 2, 20, [
        Waypoint(0, 0, 20, 0), Waypoint(200, 0, 20, 20)])
    b = make_sortie("B", 1, 20, [
        Waypoint(100, -100, 20, 0), Waypoint(100, 100, 20, 20)])
    return [a, b]


def test_real_solver_optimal_and_conflict_free():
    sorties = crossing_sorties()
    pcs = build_pair_constraints(sorties, 30.0, 10.0)
    res = run_solve(sorties, pcs, {}, time_limit=20.0)
    check(res.status == "optimal", f"small instance must prove optimal, got {res.status}")
    check(res.feasible and res.offsets, "optimal run carries a table")
    check(sum(s.weight * res.offsets[s.id] for s in sorties) == res.weighted_delay,
          "weighted_delay matches offsets")
    check(scan_schedule(sorties, res.offsets, 30.0, 10.0) == [],
          "solved table is conflict-free")
    # Locks are honoured and change the proven plan deterministically.
    res2 = run_solve(sorties, pcs, {"B": 0}, time_limit=20.0)
    check(res2.status == "optimal" and res2.offsets.get("B") == 0,
          f"lock honoured, got {res2.status} {res2.offsets}")
    check(res2.locked == {"B": 0}, "locked echoed in result")


def test_real_solver_infeasible_and_unknown_distinct():
    # A and B forced to the same take-off offset while their tracks cross:
    # window 0 for both makes the crossing unavoidable.
    a = make_sortie("A", 1, 0, [Waypoint(0, 0, 20, 0), Waypoint(200, 0, 20, 20)])
    b = make_sortie("B", 1, 0, [Waypoint(100, -100, 20, 0), Waypoint(100, 100, 20, 20)])
    sorties = [a, b]
    pcs = build_pair_constraints(sorties, 30.0, 10.0)
    res = run_solve(sorties, pcs, {}, time_limit=20.0)
    check(not res.feasible and res.status == "infeasible",
          f"proven infeasible, got {res.status}")
    check(res.cores, "infeasible carries minimal cores")
    check(any(set(c["sorties"]) == {"A", "B"} for c in res.cores),
          f"core over both sorties expected, got {res.cores}")


def test_shared_time_budget():
    # The whole staged pipeline must respect one time budget, not one per
    # stage.  Generous slack keeps this robust on loaded machines.
    sorties = crossing_sorties()
    pcs = build_pair_constraints(sorties, 30.0, 10.0)
    limit = 1.0
    res = run_solve(sorties, pcs, {}, time_limit=limit)
    check(res.elapsed_seconds <= limit + 2.0,
          f"elapsed {res.elapsed_seconds}s exceeds shared budget {limit}s")
    check(res.status in ("optimal", "feasible"),
          f"easy instance still solved, got {res.status}")


# ---------------------------------------------------------------------------
# HTTP API: timeout must not overwrite a stored plan
# ---------------------------------------------------------------------------


def test_api_timeout_keeps_stored_plan():
    os.environ["DCC_DB_PATH"] = os.path.join(
        tempfile.mkdtemp(prefix="dcc-stages-"), "console.db")
    # Import after env var is set so the app picks up the temp database.
    for mod in ("app.main",):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    import app.main as main_mod

    payload = {
        "safety_radius": 30, "safety_height": 10,
        "sorties": [
            {"id": "A", "weight": 2, "window_hi": 20,
             "waypoints": [{"x": 0, "y": 0, "z": 20, "t": 0},
                           {"x": 200, "y": 0, "z": 20, "t": 20}]},
            {"id": "B", "weight": 1, "window_hi": 20,
             "waypoints": [{"x": 100, "y": -100, "z": 20, "t": 0},
                           {"x": 100, "y": 100, "z": 20, "t": 20}]},
        ],
    }
    with TestClient(main_mod.app) as client:
        sid = client.post("/api/import", json=payload).json()["scenario_id"]

        # Real solve: proves optimal and stores the plan.
        r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20}).json()
        sol = r["solution"]
        check(sol["status"] == "optimal" and sol["feasible"],
              f"initial solve optimal, got {sol['status']}")
        good_offsets = dict(sol["offsets"])
        bid = client.post(f"/api/scenarios/{sid}/batches", json={"name": "b0"})
        check(bid.status_code == 201, f"batch from optimal plan: {bid.status_code}")

        # Timed-out re-solve (stubbed UNKNOWN): response reports the timeout,
        # but the stored plan must survive for the page and batch creation.
        original = main_mod.run_solve
        main_mod.run_solve = lambda *a, **k: SolveResult(feasible=False, status="unknown")
        try:
            r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 1}).json()
        finally:
            main_mod.run_solve = original
        check(r["solution"]["status"] == "unknown" and not r["solution"]["feasible"],
              f"timeout reported as unknown, got {r['solution']['status']}")
        stored = client.get(f"/api/scenarios/{sid}").json()["solution"]
        check(stored["status"] == "optimal" and stored["offsets"] == good_offsets,
              f"stored plan must survive a timeout, got {stored['status']}")
        again = client.post(f"/api/scenarios/{sid}/batches", json={"name": "b1"})
        check(again.status_code == 201,
              f"batch creation still possible after timeout: {again.status_code} {again.text}")

        # A witnessed-but-unproven table (feasible) IS persisted and usable.
        witness = SolveResult(feasible=True, status="feasible",
                              offsets={"A": 6, "B": 0}, weighted_delay=12)
        main_mod.run_solve = lambda *a, **k: witness
        try:
            r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 1}).json()
        finally:
            main_mod.run_solve = original
        check(r["solution"]["status"] == "feasible", "feasible status surfaced")
        stored = client.get(f"/api/scenarios/{sid}").json()["solution"]
        check(stored["status"] == "feasible" and stored["offsets"] == {"A": 6, "B": 0},
              f"feasible witness persisted, got {stored['status']} {stored['offsets']}")
        b2 = client.post(f"/api/scenarios/{sid}/batches", json={"name": "b2"})
        check(b2.status_code == 201 and b2.json()["planned_offsets"] == {"A": 6, "B": 0},
              f"batch from feasible witness: {b2.status_code} {b2.text}")

        # A proven infeasible re-solve is persisted (it is conclusive) and
        # blocks new batches until a feasible plan is solved again.
        main_mod.run_solve = lambda *a, **k: SolveResult(
            feasible=False, status="infeasible",
            cores=[{"sorties": ["A", "B"], "size": 2, "mutex_offsets": []}])
        try:
            r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 1}).json()
        finally:
            main_mod.run_solve = original
        stored = client.get(f"/api/scenarios/{sid}").json()["solution"]
        check(stored["status"] == "infeasible" and not stored["feasible"],
              f"infeasible persisted, got {stored['status']}")
        b3 = client.post(f"/api/scenarios/{sid}/batches", json={"name": "b3"})
        check(b3.status_code == 400, f"no batch from infeasible: {b3.status_code}")

        # First-ever solve being unknown IS recorded (nothing to protect).
        sid2 = client.post("/api/import", json=payload).json()["scenario_id"]
        main_mod.run_solve = lambda *a, **k: SolveResult(feasible=False, status="unknown")
        try:
            client.post(f"/api/scenarios/{sid2}/solve", json={"time_limit": 1})
        finally:
            main_mod.run_solve = original
        stored2 = client.get(f"/api/scenarios/{sid2}").json()["solution"]
        check(stored2 is not None and stored2["status"] == "unknown",
              f"first unknown result recorded, got {stored2}")


def main():
    test_stage1_unknown_falls_back_to_probe_witness()
    test_stage1_feasible_witness_replaces_only_when_not_worse()
    test_all_stages_optimal_marks_optimal()
    test_lex_feasible_round_not_locked_not_optimal()
    test_lex_unknown_round_keeps_previous_candidate()
    test_probe_unknown_stays_unknown()
    test_real_solver_optimal_and_conflict_free()
    test_real_solver_infeasible_and_unknown_distinct()
    test_shared_time_budget()
    test_api_timeout_keeps_stored_plan()
    print("failures:", FAILS)
    return FAILS


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
