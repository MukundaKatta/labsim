"""Simulation engine that solves ODEs and evaluates analytic solutions."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from labsim.models import Experiment, SimResult


class SimulationEngine:
    """Run experiments by solving their ODE systems or evaluating analytic forms."""

    def __init__(
        self,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def solve(self, experiment: Experiment) -> SimResult:
        """Execute *experiment* and return a SimResult.

        Strategy
        --------
        1. If *ode_system* is provided, integrate numerically with scipy.
        2. Else if *analytic_solution* is provided, evaluate it over t_eval.
        3. Otherwise raise.
        """
        t_eval = np.linspace(
            experiment.t_span[0],
            experiment.t_span[1],
            experiment.t_eval_points,
        )

        if experiment.ode_system is not None:
            return self._solve_ode(experiment, t_eval)

        if experiment.analytic_solution is not None:
            return self._solve_analytic(experiment, t_eval)

        raise ValueError(
            f"Experiment '{experiment.name}' has neither an ODE system "
            f"nor an analytic solution."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _solve_ode(self, exp: Experiment, t_eval: np.ndarray) -> SimResult:
        """Integrate the ODE system using scipy.integrate.solve_ivp."""

        # Event function: stop if projectile hits ground (y < 0)
        events = []
        if exp.name == "Projectile Motion":

            def hit_ground(t, state):
                return state[1]  # y component

            hit_ground.terminal = True
            hit_ground.direction = -1
            events.append(hit_ground)

        sol = solve_ivp(
            fun=exp.ode_system,
            t_span=exp.t_span,
            y0=exp.initial_conditions,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            events=events if events else None,
        )

        if not sol.success:
            raise RuntimeError(
                f"ODE solver failed for '{exp.name}': {sol.message}"
            )

        # Trim to valid range (e.g. projectile above ground)
        t = sol.t
        y_arrays = [sol.y[i] for i in range(sol.y.shape[0])]

        summary = self._compute_summary(exp, t, y_arrays)

        return SimResult(
            experiment_name=exp.name,
            discipline=exp.discipline,
            t=t,
            y=y_arrays,
            labels=exp.labels,
            summary=summary,
            metadata=exp.metadata,
        )

    def _solve_analytic(self, exp: Experiment, t_eval: np.ndarray) -> SimResult:
        """Evaluate the closed-form analytic solution."""
        result = exp.analytic_solution(t_eval)

        # If the analytic solution returns a dict (algebraic experiment)
        if isinstance(result, dict):
            return SimResult(
                experiment_name=exp.name,
                discipline=exp.discipline,
                t=t_eval,
                y=[],
                labels=exp.labels,
                summary={
                    k: v
                    for k, v in result.items()
                    if isinstance(v, (int, float))
                },
                metadata={**exp.metadata, "analytic_result": result},
            )

        # Numeric array(s)
        if isinstance(result, tuple):
            y_arrays = [np.asarray(r) for r in result]
        else:
            y_arrays = [np.asarray(result)]

        summary = self._compute_summary(exp, t_eval, y_arrays)

        return SimResult(
            experiment_name=exp.name,
            discipline=exp.discipline,
            t=t_eval,
            y=y_arrays,
            labels=exp.labels,
            summary=summary,
            metadata=exp.metadata,
        )

    @staticmethod
    def _compute_summary(
        exp: Experiment,
        t: np.ndarray,
        y_arrays: list[np.ndarray],
    ) -> dict[str, float]:
        """Derive summary statistics from simulation output."""
        summary: dict[str, float] = {}

        if exp.name == "Projectile Motion" and len(y_arrays) >= 2:
            x, y = y_arrays[0], y_arrays[1]
            summary["max_height_m"] = float(np.max(y))
            summary["range_m"] = float(x[-1]) if len(x) > 0 else 0.0
            summary["time_of_flight_s"] = float(t[-1]) if len(t) > 0 else 0.0

        elif exp.name == "Simple Pendulum" and len(y_arrays) >= 1:
            theta = y_arrays[0]
            summary["max_angle_rad"] = float(np.max(np.abs(theta)))
            if "small_angle_period" in exp.metadata:
                summary["small_angle_period_s"] = exp.metadata[
                    "small_angle_period"
                ]

        elif "Kinetics" in exp.name and len(y_arrays) >= 1:
            c = y_arrays[0]
            summary["final_concentration_M"] = float(c[-1])
            if "half_life" in exp.metadata:
                summary["half_life_s"] = exp.metadata["half_life"]

        elif "Logistic" in exp.name and len(y_arrays) >= 1:
            pop = y_arrays[0]
            summary["final_population"] = float(pop[-1])
            if "doubling_time" in exp.metadata:
                summary["doubling_time"] = exp.metadata["doubling_time"]

        elif "Lotka" in exp.name and len(y_arrays) >= 2:
            summary["prey_max"] = float(np.max(y_arrays[0]))
            summary["predator_max"] = float(np.max(y_arrays[1]))

        return summary
