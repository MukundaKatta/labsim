"""Pydantic models for experiments, results, and lab reports."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class LabDiscipline(str, Enum):
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"


class Experiment(BaseModel):
    """Describes an experiment to be run by the simulation engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    discipline: LabDiscipline
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    initial_conditions: list[float] = Field(default_factory=list)
    t_span: tuple[float, float] = (0.0, 10.0)
    t_eval_points: int = 500
    ode_system: Any = None  # callable(t, y, **params) -> dydt
    analytic_solution: Any = None  # optional closed-form callable
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Axis / variable labels, e.g. {'x': 'Horizontal distance (m)'}",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimResult(BaseModel):
    """Stores the numerical output of a simulation run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiment_name: str
    discipline: LabDiscipline
    t: Any  # np.ndarray stored as Any for pydantic compat
    y: Any  # np.ndarray or list[np.ndarray]
    labels: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert arrays to lists for serialisation."""
        return {
            "experiment_name": self.experiment_name,
            "discipline": self.discipline.value,
            "t": self.t.tolist() if isinstance(self.t, np.ndarray) else self.t,
            "y": (
                [yi.tolist() if isinstance(yi, np.ndarray) else yi for yi in self.y]
                if isinstance(self.y, list)
                else self.y.tolist() if isinstance(self.y, np.ndarray) else self.y
            ),
            "labels": self.labels,
            "summary": self.summary,
        }


class LabReport(BaseModel):
    """Structured lab report for an experiment."""

    title: str
    discipline: LabDiscipline
    date: datetime = Field(default_factory=datetime.now)
    objective: str = ""
    theory: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    results_summary: dict[str, float] = Field(default_factory=dict)
    conclusion: str = ""
    figures: list[str] = Field(
        default_factory=list,
        description="File paths to generated plots",
    )
