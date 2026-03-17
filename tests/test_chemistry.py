"""Tests for the chemistry lab simulations."""

import math

import numpy as np
import pytest

from labsim.labs.chemistry import ChemistryLab, R_GAS
from labsim.simulator.engine import SimulationEngine


@pytest.fixture
def engine():
    return SimulationEngine()


class TestIdealGas:
    def test_pressure_volume_relation(self, engine):
        """P = nRT/V - doubling volume should halve pressure."""
        exp = ChemistryLab.ideal_gas(n_moles=1.0, temperature_K=300, volume_range=(1.0, 50.0))
        result = engine.solve(exp)
        p = result.y[0]
        # Pressure at start should be higher than at end
        assert p[0] > p[-1]

    def test_known_pressure(self, engine):
        """1 mol at 273.15 K in 22.414 L should give ~101325 Pa."""
        exp = ChemistryLab.ideal_gas(n_moles=1.0, temperature_K=273.15, volume_range=(22.414, 22.415))
        result = engine.solve(exp)
        p = result.y[0][0]
        expected = 1.0 * R_GAS * 273.15 / (22.414e-3)  # ~101325 Pa
        assert abs(p - expected) / expected < 0.001

    def test_isothermal_work(self):
        """W = nRT * ln(V2/V1)."""
        exp = ChemistryLab.ideal_gas(n_moles=2.0, temperature_K=300, volume_range=(1.0, 10.0))
        expected = 2.0 * R_GAS * 300 * math.log(10.0 / 1.0)
        assert abs(exp.metadata["isothermal_work_J"] - expected) < 0.01


class TestReactionKinetics:
    def test_first_order_decay(self, engine):
        """[A] = [A]0 * exp(-kt). After one half-life, [A] ~ 0.5*[A]0."""
        k = 0.1
        c0 = 1.0
        half_life = math.log(2) / k
        exp = ChemistryLab.reaction_kinetics(order=1, k=k, initial_concentration=c0, t_max=100)
        result = engine.solve(exp)
        # Find concentration at half-life
        idx = np.argmin(np.abs(result.t - half_life))
        c_half = result.y[0][idx]
        assert abs(c_half - 0.5 * c0) < 0.01

    def test_second_order_half_life(self, engine):
        """Half-life for 2nd order: t_1/2 = 1/(k*[A]0)."""
        k, c0 = 0.05, 2.0
        exp = ChemistryLab.reaction_kinetics(order=2, k=k, initial_concentration=c0, t_max=200)
        expected_hl = 1.0 / (k * c0)
        assert abs(exp.metadata["half_life"] - expected_hl) < 1e-6

    def test_zero_order(self, engine):
        """Zero-order: [A] = [A]0 - kt (linear decrease)."""
        k, c0 = 0.1, 5.0
        exp = ChemistryLab.reaction_kinetics(order=0, k=k, initial_concentration=c0, t_max=40)
        result = engine.solve(exp)
        # At t=10, [A] should be ~4.0
        idx = np.argmin(np.abs(result.t - 10.0))
        assert abs(result.y[0][idx] - 4.0) < 0.05


class TestTitration:
    def test_equivalence_volume(self):
        """Equivalence at V_base = Ca*Va / Cb."""
        exp = ChemistryLab.titration(
            acid_concentration=0.1, acid_volume_mL=50.0, base_concentration=0.1
        )
        expected = 0.1 * 50.0 / 0.1  # 50 mL
        assert abs(exp.metadata["equivalence_volume_mL"] - expected) < 1e-6

    def test_ph_at_half_equivalence(self, engine):
        """At half-equivalence, pH = pKa."""
        ka = 1.8e-5
        exp = ChemistryLab.titration(
            acid_concentration=0.1,
            acid_volume_mL=50.0,
            base_concentration=0.1,
            ka=ka,
        )
        result = engine.solve(exp)
        pka = -math.log10(ka)
        half_eq_vol = exp.metadata["equivalence_volume_mL"] / 2.0
        idx = np.argmin(np.abs(result.t - half_eq_vol))
        ph_half = result.y[0][idx]
        assert abs(ph_half - pka) < 0.1


class TestMolecularGeometry:
    def test_tetrahedral(self, engine):
        exp = ChemistryLab.molecular_geometry(electron_domains=4, lone_pairs=0)
        result = engine.solve(exp)
        analytic = result.metadata["analytic_result"]
        assert analytic["geometry"] == "tetrahedral"
        assert abs(analytic["bond_angle_deg"] - 109.5) < 0.1

    def test_bent_water(self, engine):
        exp = ChemistryLab.molecular_geometry(electron_domains=4, lone_pairs=2)
        result = engine.solve(exp)
        analytic = result.metadata["analytic_result"]
        assert analytic["geometry"] == "bent"
        assert analytic["bonding_pairs"] == 2
