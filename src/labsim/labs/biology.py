"""Biology lab experiments.

Simulations
-----------
- Cell division / logistic growth
- Mendelian genetics (single-gene, two-gene crosses)
- Population dynamics (Lotka-Volterra predator-prey)
"""

from __future__ import annotations

import itertools
import numpy as np

from labsim.models import Experiment, LabDiscipline


class BiologyLab:
    """Factory for biology experiment descriptors."""

    discipline = LabDiscipline.BIOLOGY

    # ------------------------------------------------------------------
    # Cell division / logistic growth
    # ------------------------------------------------------------------
    @staticmethod
    def cell_division(
        initial_population: float = 100.0,
        growth_rate: float = 0.5,
        carrying_capacity: float = 10_000.0,
        t_max: float = 30.0,
    ) -> Experiment:
        """Logistic growth model for a cell population.

        dN/dt = r * N * (1 - N/K)

        where r = intrinsic growth rate, K = carrying capacity.
        """
        n0 = initial_population
        r = growth_rate
        k = carrying_capacity

        def ode(t, state, **_kw):
            n = state[0]
            return [r * n * (1.0 - n / k)]

        # Analytic solution for logistic growth
        def analytic(t):
            t = np.asarray(t)
            return k / (1.0 + ((k - n0) / n0) * np.exp(-r * t))

        doubling_time = np.log(2) / r

        return Experiment(
            name="Cell Division (Logistic Growth)",
            discipline=LabDiscipline.BIOLOGY,
            description=(
                f"Logistic growth: N0={n0}, r={r}, K={k}"
            ),
            parameters=dict(
                initial_population=n0,
                growth_rate=r,
                carrying_capacity=k,
            ),
            initial_conditions=[n0],
            t_span=(0.0, t_max),
            t_eval_points=600,
            ode_system=ode,
            analytic_solution=analytic,
            labels={
                "x": "Time (hours)",
                "y": "Population",
                "title": "Cell Division - Logistic Growth",
            },
            metadata=dict(doubling_time=doubling_time),
        )

    # ------------------------------------------------------------------
    # Mendelian genetics
    # ------------------------------------------------------------------
    @staticmethod
    def genetics(
        parent1_genotype: str = "Aa",
        parent2_genotype: str = "Aa",
    ) -> Experiment:
        """Mendelian single-gene cross using a Punnett square.

        Returns genotype and phenotype ratios.
        Supports mono-hybrid (e.g. Aa x Aa) and di-hybrid (e.g. AaBb x AaBb).
        """

        def _alleles(genotype: str) -> list[str]:
            """Split genotype into list of allele pairs, then gametes."""
            pairs = [genotype[i : i + 2] for i in range(0, len(genotype), 2)]
            gamete_lists = [[p[0], p[1]] for p in pairs]
            gametes = [
                "".join(combo) for combo in itertools.product(*gamete_lists)
            ]
            return gametes

        gametes1 = _alleles(parent1_genotype)
        gametes2 = _alleles(parent2_genotype)

        offspring: dict[str, int] = {}
        for g1 in gametes1:
            for g2 in gametes2:
                # Combine and sort each locus so Aa == aA
                genotype_parts = []
                for i in range(len(g1)):
                    pair = "".join(sorted([g1[i], g2[i]], key=lambda c: (c.lower(), c)))
                    genotype_parts.append(pair)
                gt = "".join(genotype_parts)
                offspring[gt] = offspring.get(gt, 0) + 1

        total = sum(offspring.values())
        ratios = {k: v / total for k, v in offspring.items()}

        # Phenotype: dominant if at least one uppercase allele per locus
        phenotypes: dict[str, float] = {}
        for gt, count in offspring.items():
            loci = [gt[i : i + 2] for i in range(0, len(gt), 2)]
            pheno = "-".join(
                "dominant" if any(c.isupper() for c in locus) else "recessive"
                for locus in loci
            )
            phenotypes[pheno] = phenotypes.get(pheno, 0) + count / total

        def analytic(_t):
            return {
                "offspring_genotypes": offspring,
                "genotype_ratios": ratios,
                "phenotype_ratios": phenotypes,
            }

        return Experiment(
            name="Mendelian Genetics",
            discipline=LabDiscipline.BIOLOGY,
            description=f"Cross: {parent1_genotype} x {parent2_genotype}",
            parameters=dict(
                parent1=parent1_genotype,
                parent2=parent2_genotype,
            ),
            t_span=(0.0, 1.0),
            t_eval_points=2,
            analytic_solution=analytic,
            labels={"title": "Punnett Square Analysis"},
            metadata=dict(
                genotype_ratios=ratios,
                phenotype_ratios=phenotypes,
            ),
        )

    # ------------------------------------------------------------------
    # Lotka-Volterra predator-prey
    # ------------------------------------------------------------------
    @staticmethod
    def population_dynamics(
        prey_initial: float = 40.0,
        predator_initial: float = 9.0,
        alpha: float = 0.1,
        beta: float = 0.02,
        delta: float = 0.01,
        gamma: float = 0.1,
        t_max: float = 200.0,
    ) -> Experiment:
        """Lotka-Volterra predator-prey model.

        dx/dt =  alpha * x  -  beta * x * y    (prey)
        dy/dt =  delta * x * y  -  gamma * y   (predator)

        Parameters
        ----------
        alpha : prey birth rate
        beta  : predation rate
        delta : predator reproduction per prey consumed
        gamma : predator death rate
        """

        def ode(t, state, **_kw):
            x, y = state  # prey, predator
            dx = alpha * x - beta * x * y
            dy = delta * x * y - gamma * y
            return [dx, dy]

        return Experiment(
            name="Lotka-Volterra Predator-Prey",
            discipline=LabDiscipline.BIOLOGY,
            description=(
                f"Prey0={prey_initial}, Predator0={predator_initial}, "
                f"a={alpha}, b={beta}, d={delta}, g={gamma}"
            ),
            parameters=dict(
                prey_initial=prey_initial,
                predator_initial=predator_initial,
                alpha=alpha,
                beta=beta,
                delta=delta,
                gamma=gamma,
            ),
            initial_conditions=[prey_initial, predator_initial],
            t_span=(0.0, t_max),
            t_eval_points=2000,
            ode_system=ode,
            labels={
                "y0": "Prey population",
                "y1": "Predator population",
                "x": "Time",
                "title": "Lotka-Volterra Predator-Prey Dynamics",
            },
        )
