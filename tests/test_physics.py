"""Tests for the physics lab simulations."""

import math

import numpy as np
import pytest

from labsim.labs.physics import PhysicsLab, G
from labsim.simulator.engine import SimulationEngine


@pytest.fixture
def engine():
    return SimulationEngine()


class TestProjectileMotion:
    def test_basic_launch(self, engine):
        exp = PhysicsLab.projectile_motion(velocity=20, angle_deg=45)
        result = engine.solve(exp)
        assert result.experiment_name == "Projectile Motion"
        assert len(result.t) > 0
        assert len(result.y) == 4  # x, y, vx, vy

    def test_range_45_degrees(self, engine):
        """At 45 degrees, range = v^2 / g (no drag, ground level)."""
        v = 20.0
        exp = PhysicsLab.projectile_motion(velocity=v, angle_deg=45, height=0)
        result = engine.solve(exp)
        expected_range = v**2 / G  # ~40.77 m
        # Allow 2% tolerance (numerical integration + event detection)
        assert abs(result.summary["range_m"] - expected_range) / expected_range < 0.02

    def test_max_height(self, engine):
        """Max height = (v*sin(theta))^2 / (2*g)."""
        v, angle = 30.0, 60.0
        exp = PhysicsLab.projectile_motion(velocity=v, angle_deg=angle)
        result = engine.solve(exp)
        vy0 = v * math.sin(math.radians(angle))
        expected_h = vy0**2 / (2 * G)
        assert abs(result.summary["max_height_m"] - expected_h) / expected_h < 0.01

    def test_vertical_launch(self, engine):
        exp = PhysicsLab.projectile_motion(velocity=10, angle_deg=90)
        result = engine.solve(exp)
        assert result.summary["max_height_m"] > 0
        assert result.summary["range_m"] < 0.1  # nearly zero horizontal

    def test_with_drag(self, engine):
        """Drag should reduce range compared to no-drag case."""
        no_drag = PhysicsLab.projectile_motion(velocity=20, angle_deg=45, drag_coefficient=0)
        with_drag = PhysicsLab.projectile_motion(velocity=20, angle_deg=45, drag_coefficient=0.5, mass=1.0)
        r_no = engine.solve(no_drag)
        r_drag = engine.solve(with_drag)
        assert r_drag.summary["range_m"] < r_no.summary["range_m"]


class TestPendulum:
    def test_small_angle_period(self, engine):
        """Small-angle approximation: T = 2*pi*sqrt(L/g)."""
        length = 1.0
        exp = PhysicsLab.pendulum(length=length, angle_deg=5, t_max=50)
        result = engine.solve(exp)
        expected_period = 2 * math.pi * math.sqrt(length / G)
        assert abs(result.summary["small_angle_period_s"] - expected_period) < 1e-6

    def test_oscillation(self, engine):
        exp = PhysicsLab.pendulum(length=2.0, angle_deg=15)
        result = engine.solve(exp)
        theta = result.y[0]
        # Should oscillate: crosses zero multiple times
        crossings = np.sum(np.diff(np.sign(theta)) != 0)
        assert crossings >= 4

    def test_damping_reduces_amplitude(self, engine):
        exp_undamped = PhysicsLab.pendulum(angle_deg=30, damping=0.0, t_max=30)
        exp_damped = PhysicsLab.pendulum(angle_deg=30, damping=0.3, t_max=30)
        r1 = engine.solve(exp_undamped)
        r2 = engine.solve(exp_damped)
        # Damped final amplitude should be smaller
        assert np.max(np.abs(r2.y[0][-100:])) < np.max(np.abs(r1.y[0][-100:]))


class TestOhmsLaw:
    def test_series_circuit(self, engine):
        resistances = [100, 200, 300]
        exp = PhysicsLab.ohms_law_circuit(voltage=12, resistances=resistances, configuration="series")
        result = engine.solve(exp)
        analytic = result.metadata.get("analytic_result", {})
        if not analytic:
            # Falls through to metadata
            assert result.metadata["r_total"] == 600
            assert abs(result.metadata["current"] - 0.02) < 1e-6
        else:
            assert analytic["total_resistance_ohm"] == 600
            assert abs(analytic["current_A"] - 0.02) < 1e-6

    def test_parallel_circuit(self, engine):
        resistances = [100, 200]
        exp = PhysicsLab.ohms_law_circuit(voltage=10, resistances=resistances, configuration="parallel")
        result = engine.solve(exp)
        expected_r = 1 / (1/100 + 1/200)  # ~66.67
        assert abs(result.metadata["r_total"] - expected_r) < 0.01


class TestThinLens:
    def test_converging_lens(self, engine):
        exp = PhysicsLab.thin_lens(focal_length=0.10, object_distance=0.20)
        result = engine.solve(exp)
        # 1/di = 1/f - 1/do = 10 - 5 = 5 => di = 0.20
        assert abs(result.metadata["image_distance"] - 0.20) < 1e-6
        assert abs(result.metadata["magnification"] - (-1.0)) < 1e-6

    def test_virtual_image(self, engine):
        # Object inside focal length -> virtual image
        exp = PhysicsLab.thin_lens(focal_length=0.20, object_distance=0.10)
        result = engine.solve(exp)
        assert result.metadata["image_distance"] < 0  # virtual
        assert not result.metadata["is_real"]
