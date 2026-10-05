"""Conflict model assembly and CP-SAT scheduling.

Pipeline:

1. For every sortie pair and every segment pair, the analytic geometry in
   :mod:`app.geometry` yields the real interval of relative take-off offsets
   ``d = t0_B - t0_A`` that produce a conflict.  The intervals are converted
   to integer sets and merged into the pair's forbidden-offset table.

2. A CP-SAT model chooses one integer take-off offset ``x_i`` per sortie
   (earliest take-off is 0, latest is the sortie's window end), disjunctively
   avoiding every forbidden interval.  Solving proceeds in stages that share
   one time budget: a feasibility probe, minimisation of the weighted total
   delay (relative to earliest take-off), then lexicographic minimisation of
   the offsets ordered by sortie ID.

   Every stage keeps the best *witnessed* take-off table consistent with the
   constraints proven so far; a new witness replaces it only when its
   weighted delay is no worse.  A stage that times out therefore falls back
   to the last witnessed table instead of discarding it.  The result is
   marked ``optimal`` only when the weighted objective AND every
   lexicographic round were proven OPTIMAL; a witnessed but unproven table
   is reported as ``feasible``, and a probe that witnesses nothing within
   the budget stays ``unknown`` — distinct from proven ``infeasible``.

3. On infeasibility, every *sortie* carries a participation boolean (assumed
   true); each pair's avoidance clause is enabled only when both sorties
   participate.  CP-SAT then returns an unsat core over these booleans, i.e.
   an inclusion-minimal ("包含关系极小") infeasible sortie set.  Further cores
   are found by branching: force one sortie of a known core out of the
   scenario and solve again; the positive literals of every such core are
   genuine inclusion-minimal infeasible sets of the original problem.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .geometry import (
    conflicts_between_sorties,
    forbidden_offset_interval_segments,
    integer_forbidden_interval,
    merge_intervals,
)


@dataclass
class PairConstraint:
    sortie_a: str
    sortie_b: str
    intervals: list[tuple[int, int]]  # merged forbidden d = tB - tA intervals


@dataclass
class SolveResult:
    feasible: bool
    status: str  # optimal | feasible | infeasible | unknown | invalid
    offsets: dict[str, int] = field(default_factory=dict)
    weighted_delay: int | None = None
    cores: list[dict] = field(default_factory=list)  # minimal unsat sets
    pair_constraints: list[PairConstraint] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    locked: dict[str, int] = field(default_factory=dict)
    solved_conflicts: list[dict] = field(default_factory=list)
    truncated: bool = False


# ---------------------------------------------------------------------------
# Build pair constraints from geometry
# ---------------------------------------------------------------------------


def build_pair_constraints(sorties, safety_radius: float, safety_height: float) -> list[PairConstraint]:
    out: list[PairConstraint] = []
    n = len(sorties)
    for i in range(n):
        sa = sorties[i]
        for j in range(i + 1, n):
            sb = sorties[j]
            raw: list[tuple[int, int]] = []
            for sega in sa.segments:
                for segb in sb.segments:
                    interval = forbidden_offset_interval_segments(
                        sega, segb, safety_radius, safety_height
                    )
                    if interval is None:
                        continue
                    ii = integer_forbidden_interval(interval)
                    if ii is not None:
                        raw.append(ii)
            if not raw:
                continue
            d_min = sb.window_lo - sa.window_hi
            d_max = sb.window_hi - sa.window_lo
            clipped = [
                (max(lo, d_min), min(hi, d_max))
                for lo, hi in raw
                if hi >= d_min and lo <= d_max
            ]
            clipped = [(lo, hi) for lo, hi in clipped if lo <= hi]
            merged = merge_intervals(clipped)
            if merged:
                out.append(PairConstraint(sa.id, sb.id, merged))
    return out


# ---------------------------------------------------------------------------
# Fixed-schedule conflict scan
# ---------------------------------------------------------------------------


def scan_schedule(sorties, offsets: dict[str, int], safety_radius: float,
                  safety_height: float) -> list[dict]:
    records: list[dict] = []
    for i in range(len(sorties)):
        sa = sorties[i]
        for j in range(i + 1, len(sorties)):
            sb = sorties[j]
            recs = conflicts_between_sorties(
                sa, offsets.get(sa.id, 0),
                sb, offsets.get(sb.id, 0),
                safety_radius, safety_height,
            )
            for r in recs:
                records.append(_record_to_dict(r))
    records.sort(key=lambda r: (r["time_start"], r["sortie_a"], r["sortie_b"]))
    return records


def _record_to_dict(r) -> dict:
    return {
        "sortie_a": r.sortie_a,
        "sortie_b": r.sortie_b,
        "time_start": round(r.time_start, 6),
        "time_end": round(r.time_end, 6),
        "witness_time": round(r.witness_time, 6),
        "a_pos": [round(v, 6) for v in r.a_pos],
        "b_pos": [round(v, 6) for v in r.b_pos],
        "horizontal_distance": round(r.horizontal_distance, 6),
        "vertical_distance": round(r.vertical_distance, 6),
    }


# ---------------------------------------------------------------------------
# CP-SAT model
# ---------------------------------------------------------------------------


def _build_model(sorties, pair_constraints: list[PairConstraint], locks: dict[str, int]):
    model = cp_model.CpModel()
    takeoff: dict[str, cp_model.IntVar] = {}
    participate: dict[str, cp_model.IntVar] = {}
    for s in sorties:
        takeoff[s.id] = model.NewIntVar(s.window_lo, s.window_hi, f"to_{s.id}")
        participate[s.id] = model.NewBoolVar(f"in_{s.id}")
        if s.id in locks:
            model.Add(takeoff[s.id] == int(locks[s.id]))

    for pc in pair_constraints:
        xa, xb = takeoff[pc.sortie_a], takeoff[pc.sortie_b]
        pa, pb = participate[pc.sortie_a], participate[pc.sortie_b]
        for k, (lo, hi) in enumerate(pc.intervals):
            b_after = model.NewBoolVar(f"after_{pc.sortie_a}_{pc.sortie_b}_{k}")
            b_before = model.NewBoolVar(f"before_{pc.sortie_a}_{pc.sortie_b}_{k}")
            model.Add(xb - xa >= hi + 1).OnlyEnforceIf(b_after)
            model.Add(xb - xa <= lo - 1).OnlyEnforceIf(b_before)
            # pair constraint active only while both sorties participate
            model.AddBoolOr([b_after, b_before]).OnlyEnforceIf(pa, pb)
    return model, takeoff, participate


def _new_solver(time_limit: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    return solver


# Minimum budget handed to a solve stage; below this the stage is skipped and
# the current candidate is returned instead of gambling on a ~0-second solve.
_MIN_STAGE_SECONDS = 1e-3


def run_solve(sorties, pair_constraints: list[PairConstraint],
              locks: dict[str, int] | None = None,
              time_limit: float = 30.0, max_cores: int = 40) -> SolveResult:
    locks = dict(locks or {})
    t_start = time.perf_counter()
    deadline = t_start + float(time_limit)
    model, takeoff, participate = _build_model(sorties, pair_constraints, locks)
    idx_to_id = {v.Index(): sid for sid, v in participate.items()}
    weights = {s.id: s.weight for s in sorties}

    def remaining() -> float:
        return deadline - time.perf_counter()

    # Best witnessed take-off table consistent with every constraint proven
    # so far.  A fresh witness replaces it only when its weighted delay is no
    # worse, so a later stage can never silently drop a better table.
    candidate_offsets: dict[str, int] | None = None
    candidate_delay: int | None = None

    def adopt(values: dict[str, int]) -> None:
        nonlocal candidate_offsets, candidate_delay
        delay = sum(weights[sid] * values[sid] for sid in weights)
        if candidate_delay is None or delay <= candidate_delay:
            candidate_offsets = dict(values)
            candidate_delay = delay

    # Stage 0: feasibility probe with all sorties participating.
    solver = _new_solver(max(remaining(), _MIN_STAGE_SECONDS))
    model.AddAssumptions([participate[sid] for sid in sorted(participate)])
    status = solver.Solve(model)

    if status == cp_model.INFEASIBLE:
        cores, truncated = _enumerate_cores(
            solver, model, participate, idx_to_id, pair_constraints,
            max_cores, t_start, time_limit,
        )
        return SolveResult(
            feasible=False,
            status="infeasible",
            cores=cores,
            pair_constraints=pair_constraints,
            elapsed_seconds=time.perf_counter() - t_start,
            locked=locks,
            truncated=truncated,
        )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No take-off table witnessed within the budget: genuinely unknown,
        # not disproven.
        return SolveResult(
            feasible=False,
            status=solver.StatusName(status).lower(),
            pair_constraints=pair_constraints,
            elapsed_seconds=time.perf_counter() - t_start,
            locked=locks,
        )
    adopt({sid: solver.Value(v) for sid, v in takeoff.items()})

    # Stage 1: minimise weighted total delay (relative to earliest take-off).
    objective_proven = False
    if remaining() > _MIN_STAGE_SECONDS:
        solver = _new_solver(remaining())
        model.Minimize(sum(weights[sid] * v for sid, v in takeoff.items()))
        st = solver.Solve(model)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            adopt({sid: solver.Value(v) for sid, v in takeoff.items()})
            objective_proven = st == cp_model.OPTIMAL

    # Stage 2: lexicographically smallest offset sequence (sortie ID order).
    # Restricting to the stage-1 objective value is only sound once that
    # value is proven optimal, and each per-sortie offset is locked only when
    # its own minimisation was proven OPTIMAL as well.
    lex_proven = objective_proven
    if objective_proven:
        model.Add(sum(weights[sid] * v for sid, v in takeoff.items())
                  == candidate_delay)
        for sid in sorted(takeoff):
            if remaining() <= _MIN_STAGE_SECONDS:
                lex_proven = False
                break
            solver = _new_solver(remaining())
            model.Minimize(takeoff[sid])
            st = solver.Solve(model)
            if st == cp_model.OPTIMAL:
                model.Add(takeoff[sid] == solver.Value(takeoff[sid]))
                adopt({k: solver.Value(v) for k, v in takeoff.items()})
                continue
            # FEASIBLE: the witnessed table is valid but the offset is not
            # proven minimal — adopt it without locking; UNKNOWN: keep the
            # previous candidate.  Either way the lex proof stays incomplete.
            if st == cp_model.FEASIBLE:
                adopt({k: solver.Value(v) for k, v in takeoff.items()})
            lex_proven = False
            break

    return SolveResult(
        feasible=True,
        status="optimal" if (objective_proven and lex_proven) else "feasible",
        offsets=candidate_offsets,
        weighted_delay=candidate_delay,
        pair_constraints=pair_constraints,
        elapsed_seconds=time.perf_counter() - t_start,
        locked=locks,
    )


def _extract_core_ids(solver, idx_to_id) -> tuple[list[str], list[str]]:
    """Return (participating ids, forced-out ids) of the current unsat core."""
    lits = solver.SufficientAssumptionsForInfeasibility()
    inside, outside = [], []
    for lit in lits:
        sid = idx_to_id.get(abs(lit))
        if sid is None:
            continue
        (inside if lit > 0 else outside).append(sid)
    return sorted(inside), sorted(outside)


def _enumerate_cores(solver, model, participate, idx_to_id, pair_constraints,
                     max_cores, t_start, time_limit) -> tuple[list[dict], bool]:
    found: list[frozenset[str]] = []
    queue: deque[frozenset[str]] = deque()

    first, _ = _extract_core_ids(solver, idx_to_id)
    if not first:
        return [], False
    found.append(frozenset(first))
    queue.append(frozenset(first))

    deadline = t_start + time_limit
    while queue and len(found) < max_cores and time.perf_counter() < deadline:
        core = queue.popleft()
        # Branch: drop each member of this core one at a time; a fresh
        # infeasibility yields a different minimal core not containing it.
        members = sorted(core)
        for dropped in members:
            if len(found) >= max_cores or time.perf_counter() >= deadline:
                break
            assm = []
            for sid, var in participate.items():
                assm.append(var.Not() if sid == dropped else var)
            model.ClearAssumptions()
            model.AddAssumptions(assm)
            st = solver.Solve(model)
            if st != cp_model.INFEASIBLE:
                continue
            inside, _outside = _extract_core_ids(solver, idx_to_id)
            if not inside:
                continue
            key = frozenset(inside)
            if key not in found:
                found.append(key)
                queue.append(key)

    model.ClearAssumptions()
    model.AddAssumptions([participate[sid] for sid in sorted(participate)])

    pair_index = {(pc.sortie_a, pc.sortie_b): pc.intervals for pc in pair_constraints}
    cores_out = []
    for key in sorted(found, key=lambda s: (len(s), sorted(s))):
        ids = sorted(key)
        mutex = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ivs = pair_index.get((a, b))
                if ivs:
                    mutex.append({
                        "sortie_a": a,
                        "sortie_b": b,
                        "forbidden_offsets": [[lo, hi] for lo, hi in ivs],
                    })
        cores_out.append({"sorties": ids, "size": len(ids), "mutex_offsets": mutex})
    truncated = bool(queue) or len(found) >= max_cores
    return cores_out, truncated
