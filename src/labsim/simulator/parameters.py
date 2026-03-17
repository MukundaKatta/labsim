"""Configurable experiment parameters with validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExperimentParameters(BaseModel):
    """User-facing configuration for any experiment.

    Provides validation and conversion utilities so the CLI can
    accept key=value pairs and pass them into lab factories.
    """

    lab: str = Field(description="Lab discipline: physics, chemistry, biology")
    experiment: str = Field(description="Experiment name, e.g. 'projectile', 'pendulum'")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value experiment parameters",
    )
    solver_method: str = "RK45"
    t_max: float | None = None
    output_format: str = Field(
        default="rich",
        description="Output format: rich, json, csv",
    )
    save_plot: str | None = Field(
        default=None,
        description="File path to save the plot (e.g. 'result.png')",
    )

    @field_validator("lab")
    @classmethod
    def validate_lab(cls, v: str) -> str:
        v = v.lower().strip()
        allowed = {"physics", "chemistry", "biology"}
        if v not in allowed:
            raise ValueError(f"Lab must be one of {allowed}, got '{v}'")
        return v

    @field_validator("experiment")
    @classmethod
    def validate_experiment(cls, v: str) -> str:
        return v.lower().strip().replace("-", "_").replace(" ", "_")

    # ------------------------------------------------------------------
    # Registry of known experiments and their default parameters
    # ------------------------------------------------------------------
    EXPERIMENT_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
        "physics": {
            "projectile": dict(velocity=20.0, angle_deg=45.0, height=0.0, drag_coefficient=0.0),
            "pendulum": dict(length=1.0, angle_deg=30.0, damping=0.0),
            "circuit": dict(voltage=12.0, resistances=[100, 220, 330], configuration="series"),
            "lens": dict(focal_length=0.15, object_distance=0.30, object_height=0.05),
        },
        "chemistry": {
            "ideal_gas": dict(n_moles=1.0, temperature_K=298.15),
            "kinetics": dict(order=1, k=0.1, initial_concentration=1.0),
            "titration": dict(acid_concentration=0.1, acid_volume_mL=50.0, base_concentration=0.1),
            "molecular": dict(electron_domains=4, lone_pairs=0),
        },
        "biology": {
            "cell_division": dict(initial_population=100, growth_rate=0.5, carrying_capacity=10000),
            "genetics": dict(parent1_genotype="Aa", parent2_genotype="Aa"),
            "population": dict(prey_initial=40, predator_initial=9, alpha=0.1, beta=0.02, delta=0.01, gamma=0.1),
        },
    }

    def get_merged_params(self) -> dict[str, Any]:
        """Merge user params over the defaults for the chosen experiment."""
        defaults = (
            self.EXPERIMENT_REGISTRY.get(self.lab, {})
            .get(self.experiment, {})
            .copy()
        )
        defaults.update(self.params)
        if self.t_max is not None:
            defaults["t_max"] = self.t_max
        return defaults

    @classmethod
    def list_experiments(cls, lab: str | None = None) -> dict[str, list[str]]:
        """Return available experiments, optionally filtered by lab."""
        if lab:
            return {lab: list(cls.EXPERIMENT_REGISTRY.get(lab, {}).keys())}
        return {k: list(v.keys()) for k, v in cls.EXPERIMENT_REGISTRY.items()}
