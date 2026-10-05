"""Pydantic schemas and import-time validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class WaypointIn(BaseModel):
    x: float
    y: float
    z: float
    t: int = Field(..., description="seconds from take-off; strictly increasing, >= 0")

    @field_validator("t")
    @classmethod
    def t_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("waypoint time must be >= 0")
        return v


class SortieIn(BaseModel):
    id: str = Field(..., min_length=1)
    weight: int = Field(..., gt=0, description="positive integer priority weight")
    window_hi: int = Field(..., ge=0, description="latest take-off offset in seconds")
    waypoints: list[WaypointIn] = Field(..., min_length=2)

    @field_validator("id")
    @classmethod
    def id_nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sortie id must not be blank")
        return v

    @model_validator(mode="after")
    def _check(self):
        ts = [w.t for w in self.waypoints]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("waypoint times must be strictly increasing integers")
        if self.window_hi < 0:
            raise ValueError("window_hi must be >= 0")
        return self


class ImportPayload(BaseModel):
    sorties: list[SortieIn] = Field(..., min_length=1)
    safety_radius: float = Field(..., gt=0)
    safety_height: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [s.id for s in self.sorties]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate sortie ids: {dup}")
        return self


class LockEntry(BaseModel):
    sortie_id: str
    offset: int = Field(..., ge=0)


class SolveRequest(BaseModel):
    safety_radius: float | None = Field(None, gt=0)
    safety_height: float | None = Field(None, gt=0)
    locks: list[LockEntry] = Field(default_factory=list)
    time_limit: float = Field(20.0, gt=0, le=120)


class SaveRequest(BaseModel):
    name: str = Field(..., min_length=1)


class CreateBatchRequest(BaseModel):
    name: str = Field("", description="optional display name for the batch")


class ReportRequest(BaseModel):
    sortie_id: str = Field(..., min_length=1)
    actual_offset: int = Field(
        ..., ge=0, description="actual take-off offset in seconds (non-negative)")
    revision: int = Field(
        ..., ge=1, description="report revision, 1 first then strictly increasing")
