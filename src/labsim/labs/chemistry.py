"""Chemistry lab experiments.

Simulations
-----------
- Ideal gas law (PV = nRT)
- Reaction kinetics (first-order, second-order, Michaelis-Menten)
- Acid-base titration curve
- Simple molecular geometry (VSEPR bond angles)
"""

from __future__ import annotations

import numpy as np

from labsim.models import Experiment, LabDiscipline

R_GAS = 8.314462  # J/(mol*K)


class ChemistryLab:
    """Factory for chemistry experiment descriptors."""

    discipline = LabDiscipline.CHEMISTRY

    # ------------------------------------------------------------------
    # Ideal gas law
    # ------------------------------------------------------------------
    @staticmethod
    def ideal_gas(
        n_moles: float = 1.0,
        temperature_K: float = 298.15,
        volume_range: tuple[float, float] = (0.5, 50.0),
        points: int = 500,
    ) -> Experiment:
        """PV = nRT  =>  P = nRT / V.

        Generates P(V) curve (isothermal) and computes work for
        isothermal expansion: W = nRT * ln(V2/V1).
        """
        nrt = n_moles * R_GAS * temperature_K
        work = nrt * np.log(volume_range[1] / volume_range[0])

        def analytic(v):
            """Pressure as a function of volume (L converted to m^3 internally)."""
            v_m3 = np.asarray(v) * 1e-3  # L -> m^3
            return nrt / v_m3  # Pa

        return Experiment(
            name="Ideal Gas Law",
            discipline=LabDiscipline.CHEMISTRY,
            description=(
                f"Isothermal curve: n={n_moles} mol, T={temperature_K} K"
            ),
            parameters=dict(
                n_moles=n_moles,
                temperature_K=temperature_K,
                R=R_GAS,
                volume_range_L=volume_range,
            ),
            t_span=volume_range,
            t_eval_points=points,
            analytic_solution=analytic,
            labels={
                "x": "Volume (L)",
                "y": "Pressure (Pa)",
                "title": "Ideal Gas Isotherm",
            },
            metadata=dict(
                isothermal_work_J=work,
                nrt=nrt,
            ),
        )

    # ------------------------------------------------------------------
    # Reaction kinetics
    # ------------------------------------------------------------------
    @staticmethod
    def reaction_kinetics(
        order: int = 1,
        k: float = 0.1,
        initial_concentration: float = 1.0,
        t_max: float = 50.0,
    ) -> Experiment:
        """Reaction A -> products.

        0th order: d[A]/dt = -k                => [A] = [A]0 - kt
        1st order: d[A]/dt = -k[A]             => [A] = [A]0 * exp(-kt)
        2nd order: d[A]/dt = -k[A]^2           => 1/[A] = 1/[A]0 + kt

        Half-life:
            0th: t_1/2 = [A]0 / (2k)
            1st: t_1/2 = ln(2) / k
            2nd: t_1/2 = 1 / (k * [A]0)
        """
        c0 = initial_concentration

        def ode(t, state, **_kw):
            c = max(state[0], 0.0)
            if order == 0:
                return [-k]
            elif order == 1:
                return [-k * c]
            else:
                return [-k * c**2]

        if order == 0:
            half_life = c0 / (2.0 * k)
        elif order == 1:
            half_life = np.log(2) / k
        else:
            half_life = 1.0 / (k * c0)

        return Experiment(
            name=f"Reaction Kinetics (order {order})",
            discipline=LabDiscipline.CHEMISTRY,
            description=(
                f"{order}-order reaction, k={k}, [A]0={c0} M"
            ),
            parameters=dict(order=order, k=k, initial_concentration=c0),
            initial_conditions=[c0],
            t_span=(0.0, t_max),
            t_eval_points=600,
            ode_system=ode,
            labels={
                "x": "Time (s)",
                "y": "Concentration (M)",
                "title": f"Reaction Kinetics (order {order})",
            },
            metadata=dict(half_life=half_life),
        )

    # ------------------------------------------------------------------
    # Acid-base titration
    # ------------------------------------------------------------------
    @staticmethod
    def titration(
        acid_concentration: float = 0.1,
        acid_volume_mL: float = 50.0,
        base_concentration: float = 0.1,
        base_volume_max_mL: float = 100.0,
        ka: float = 1.8e-5,
        points: int = 500,
    ) -> Experiment:
        """Weak-acid / strong-base titration.

        Henderson-Hasselbalch: pH = pKa + log([A-]/[HA])

        Before equivalence: buffer region
        At equivalence:     hydrolysis of conjugate base
        After equivalence:  excess OH-
        """
        pka = -np.log10(ka)
        equiv_vol = acid_concentration * acid_volume_mL / base_concentration
        kw = 1e-14

        def analytic(vb_arr):
            """pH as function of base volume added (mL)."""
            vb = np.asarray(vb_arr, dtype=float)
            ph = np.empty_like(vb)
            va = acid_volume_mL
            ca = acid_concentration
            cb = base_concentration

            for i, v in enumerate(vb):
                total_vol = va + v  # mL
                moles_acid = ca * va / 1000.0
                moles_base = cb * v / 1000.0

                if v < 1e-12:
                    # Initial pH of weak acid: [H+] = sqrt(Ka * Ca)
                    h = np.sqrt(ka * ca)
                    ph[i] = -np.log10(h)
                elif abs(moles_base - moles_acid) < 1e-12:
                    # Equivalence point
                    c_conj = moles_acid / (total_vol / 1000.0)
                    kb = kw / ka
                    oh = np.sqrt(kb * c_conj)
                    ph[i] = 14.0 + np.log10(oh)
                elif moles_base < moles_acid:
                    # Buffer region (Henderson-Hasselbalch)
                    ha = moles_acid - moles_base
                    a_minus = moles_base
                    ph[i] = pka + np.log10(a_minus / ha)
                else:
                    # Excess base
                    excess_oh = (moles_base - moles_acid) / (total_vol / 1000.0)
                    ph[i] = 14.0 + np.log10(excess_oh)
            return ph

        return Experiment(
            name="Acid-Base Titration",
            discipline=LabDiscipline.CHEMISTRY,
            description=(
                f"Weak acid (Ka={ka:.2e}, {acid_concentration} M, "
                f"{acid_volume_mL} mL) titrated with "
                f"{base_concentration} M strong base"
            ),
            parameters=dict(
                acid_concentration=acid_concentration,
                acid_volume_mL=acid_volume_mL,
                base_concentration=base_concentration,
                ka=ka,
                pka=pka,
            ),
            t_span=(0.01, base_volume_max_mL),
            t_eval_points=points,
            analytic_solution=analytic,
            labels={
                "x": "Volume of base added (mL)",
                "y": "pH",
                "title": "Acid-Base Titration Curve",
            },
            metadata=dict(
                equivalence_volume_mL=equiv_vol,
                pka=pka,
            ),
        )

    # ------------------------------------------------------------------
    # Molecular modeling (VSEPR bond angles)
    # ------------------------------------------------------------------
    @staticmethod
    def molecular_geometry(
        electron_domains: int = 4,
        lone_pairs: int = 0,
    ) -> Experiment:
        """VSEPR model for molecular geometry.

        Maps (electron_domains, lone_pairs) to ideal bond angle and
        geometry name.
        """
        vsepr_table: dict[tuple[int, int], tuple[str, float]] = {
            (2, 0): ("linear", 180.0),
            (3, 0): ("trigonal planar", 120.0),
            (3, 1): ("bent", 117.0),
            (4, 0): ("tetrahedral", 109.5),
            (4, 1): ("trigonal pyramidal", 107.0),
            (4, 2): ("bent", 104.5),
            (5, 0): ("trigonal bipyramidal", 90.0),
            (5, 1): ("seesaw", 86.0),
            (5, 2): ("T-shaped", 87.5),
            (5, 3): ("linear", 180.0),
            (6, 0): ("octahedral", 90.0),
            (6, 1): ("square pyramidal", 85.0),
            (6, 2): ("square planar", 90.0),
        }

        key = (electron_domains, lone_pairs)
        geometry, angle = vsepr_table.get(key, ("unknown", 0.0))
        bonding_pairs = electron_domains - lone_pairs

        def analytic(_t):
            return {
                "geometry": geometry,
                "bond_angle_deg": angle,
                "bonding_pairs": bonding_pairs,
                "lone_pairs": lone_pairs,
            }

        return Experiment(
            name="Molecular Geometry (VSEPR)",
            discipline=LabDiscipline.CHEMISTRY,
            description=(
                f"VSEPR: {electron_domains} electron domains, "
                f"{lone_pairs} lone pairs -> {geometry}"
            ),
            parameters=dict(
                electron_domains=electron_domains,
                lone_pairs=lone_pairs,
            ),
            t_span=(0.0, 1.0),
            t_eval_points=2,
            analytic_solution=analytic,
            labels={"title": "Molecular Geometry (VSEPR)"},
            metadata=dict(
                geometry=geometry,
                bond_angle_deg=angle,
                bonding_pairs=bonding_pairs,
            ),
        )
