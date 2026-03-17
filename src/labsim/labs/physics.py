"""Physics lab experiments with real equations.

Simulations
-----------
- Projectile motion (with optional drag)
- Simple pendulum (non-linear ODE)
- Ohm's law DC circuits
- Thin-lens optics
"""

from __future__ import annotations

import numpy as np

from labsim.models import Experiment, LabDiscipline

G = 9.80665  # m/s^2


class PhysicsLab:
    """Factory that builds Experiment descriptors for physics simulations."""

    discipline = LabDiscipline.PHYSICS

    # ------------------------------------------------------------------
    # Projectile motion
    # ------------------------------------------------------------------
    @staticmethod
    def projectile_motion(
        velocity: float = 20.0,
        angle_deg: float = 45.0,
        height: float = 0.0,
        drag_coefficient: float = 0.0,
        mass: float = 1.0,
        t_max: float = 10.0,
    ) -> Experiment:
        """2-D projectile with optional quadratic drag.

        State vector: [x, y, vx, vy]

        Equations of motion
        --------------------
        Without drag:
            ax = 0,  ay = -g

        With drag (quadratic):
            F_drag = -0.5 * Cd * v * v_hat
            a = F_drag / m + g_vec
        """
        theta = np.radians(angle_deg)
        vx0 = velocity * np.cos(theta)
        vy0 = velocity * np.sin(theta)
        cd = drag_coefficient

        def ode(t, state, **_kw):
            _x, y, vx, vy = state
            speed = np.sqrt(vx**2 + vy**2) + 1e-12
            drag_x = -0.5 * cd * speed * vx / mass
            drag_y = -0.5 * cd * speed * vy / mass
            return [vx, vy, drag_x, drag_y - G]

        # Analytic (no drag) for comparison
        def analytic(t):
            x = vx0 * t
            y_val = height + vy0 * t - 0.5 * G * t**2
            return x, y_val

        return Experiment(
            name="Projectile Motion",
            discipline=LabDiscipline.PHYSICS,
            description=(
                f"Projectile launched at {velocity} m/s, {angle_deg} deg "
                f"from height {height} m"
            ),
            parameters=dict(
                velocity=velocity,
                angle_deg=angle_deg,
                height=height,
                drag_coefficient=cd,
                mass=mass,
            ),
            initial_conditions=[0.0, height, vx0, vy0],
            t_span=(0.0, t_max),
            t_eval_points=800,
            ode_system=ode,
            analytic_solution=analytic if cd == 0 else None,
            labels={
                "x": "Horizontal distance (m)",
                "y": "Vertical distance (m)",
                "title": "Projectile Motion",
            },
        )

    # ------------------------------------------------------------------
    # Simple pendulum (non-linear)
    # ------------------------------------------------------------------
    @staticmethod
    def pendulum(
        length: float = 1.0,
        angle_deg: float = 30.0,
        damping: float = 0.0,
        t_max: float = 20.0,
    ) -> Experiment:
        r"""Non-linear simple pendulum.

        ODE: d^2\theta/dt^2 = -(g/L)*sin(\theta) - b*d\theta/dt

        State: [\theta, \omega]
        """
        theta0 = np.radians(angle_deg)

        def ode(t, state, **_kw):
            theta, omega = state
            dtheta = omega
            domega = -(G / length) * np.sin(theta) - damping * omega
            return [dtheta, domega]

        period_small = 2 * np.pi * np.sqrt(length / G)

        return Experiment(
            name="Simple Pendulum",
            discipline=LabDiscipline.PHYSICS,
            description=(
                f"Pendulum L={length} m, initial angle={angle_deg} deg, "
                f"damping={damping}"
            ),
            parameters=dict(
                length=length,
                angle_deg=angle_deg,
                damping=damping,
                g=G,
            ),
            initial_conditions=[theta0, 0.0],
            t_span=(0.0, t_max),
            t_eval_points=1000,
            ode_system=ode,
            labels={
                "y0": "Angle (rad)",
                "y1": "Angular velocity (rad/s)",
                "title": "Simple Pendulum",
            },
            metadata={"small_angle_period": period_small},
        )

    # ------------------------------------------------------------------
    # Ohm's law circuit
    # ------------------------------------------------------------------
    @staticmethod
    def ohms_law_circuit(
        voltage: float = 12.0,
        resistances: list[float] | None = None,
        configuration: str = "series",
    ) -> Experiment:
        """DC circuit analysis using Ohm's law: V = IR.

        For series:   R_total = R1 + R2 + ...
        For parallel: 1/R_total = 1/R1 + 1/R2 + ...

        Returns an Experiment whose analytic_solution computes I, V_drops, P.
        No ODE needed - purely algebraic.
        """
        if resistances is None:
            resistances = [100.0, 220.0, 330.0]

        if configuration == "series":
            r_total = sum(resistances)
        else:
            r_total = 1.0 / sum(1.0 / r for r in resistances)

        current = voltage / r_total
        power = voltage * current

        if configuration == "series":
            v_drops = [current * r for r in resistances]
        else:
            v_drops = [voltage] * len(resistances)

        i_branches = (
            [current] * len(resistances)
            if configuration == "series"
            else [voltage / r for r in resistances]
        )

        def analytic(_t):
            """Return circuit quantities (not time-dependent)."""
            return {
                "total_resistance_ohm": r_total,
                "current_A": current,
                "voltage_drops_V": v_drops,
                "branch_currents_A": i_branches,
                "total_power_W": power,
            }

        return Experiment(
            name="Ohm's Law Circuit",
            discipline=LabDiscipline.PHYSICS,
            description=(
                f"{configuration.title()} circuit: V={voltage} V, "
                f"R={resistances} ohm"
            ),
            parameters=dict(
                voltage=voltage,
                resistances=resistances,
                configuration=configuration,
            ),
            t_span=(0.0, 1.0),
            t_eval_points=2,
            analytic_solution=analytic,
            labels={"title": "Ohm's Law Circuit Analysis"},
            metadata=dict(
                r_total=r_total,
                current=current,
                power=power,
                v_drops=v_drops,
                i_branches=i_branches,
            ),
        )

    # ------------------------------------------------------------------
    # Thin-lens optics
    # ------------------------------------------------------------------
    @staticmethod
    def thin_lens(
        focal_length: float = 0.15,
        object_distance: float = 0.30,
        object_height: float = 0.05,
    ) -> Experiment:
        """Thin-lens equation: 1/f = 1/do + 1/di.

        Magnification: M = -di / do
        """
        di = 1.0 / (1.0 / focal_length - 1.0 / object_distance)
        magnification = -di / object_distance
        image_height = magnification * object_height
        is_real = di > 0

        def analytic(_t):
            return {
                "image_distance_m": di,
                "magnification": magnification,
                "image_height_m": image_height,
                "is_real_image": is_real,
                "is_inverted": magnification < 0,
            }

        return Experiment(
            name="Thin Lens Optics",
            discipline=LabDiscipline.PHYSICS,
            description=(
                f"f={focal_length} m, do={object_distance} m, "
                f"h_obj={object_height} m"
            ),
            parameters=dict(
                focal_length=focal_length,
                object_distance=object_distance,
                object_height=object_height,
            ),
            t_span=(0.0, 1.0),
            t_eval_points=2,
            analytic_solution=analytic,
            labels={"title": "Thin Lens Optics"},
            metadata=dict(
                image_distance=di,
                magnification=magnification,
                image_height=image_height,
                is_real=is_real,
            ),
        )
