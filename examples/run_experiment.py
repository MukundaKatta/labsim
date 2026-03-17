#!/usr/bin/env python3
"""Example: run several LABSIM experiments and display results."""

from labsim.labs.physics import PhysicsLab
from labsim.labs.chemistry import ChemistryLab
from labsim.labs.biology import BiologyLab
from labsim.simulator.engine import SimulationEngine
from labsim.simulator.visualizer import ResultVisualizer
from labsim.report import generate_report, print_report, print_result_table

from rich.console import Console

console = Console()
engine = SimulationEngine()
viz = ResultVisualizer()


def run_projectile():
    console.rule("[bold blue]Projectile Motion")
    exp = PhysicsLab.projectile_motion(velocity=25, angle_deg=60)
    result = engine.solve(exp)
    print_result_table(result, console)
    viz.plot(result, save_path="projectile.png")
    console.print("[green]Saved projectile.png[/green]\n")


def run_pendulum():
    console.rule("[bold blue]Simple Pendulum")
    exp = PhysicsLab.pendulum(length=1.5, angle_deg=45, damping=0.1)
    result = engine.solve(exp)
    print_result_table(result, console)
    viz.plot(result, save_path="pendulum.png")
    console.print("[green]Saved pendulum.png[/green]\n")


def run_circuit():
    console.rule("[bold blue]Ohm's Law Circuit")
    exp = PhysicsLab.ohms_law_circuit(voltage=9, resistances=[100, 220, 470])
    result = engine.solve(exp)
    print_result_table(result, console)


def run_titration():
    console.rule("[bold blue]Acid-Base Titration")
    exp = ChemistryLab.titration(
        acid_concentration=0.1,
        acid_volume_mL=50,
        base_concentration=0.1,
        ka=1.8e-5,
    )
    result = engine.solve(exp)
    print_result_table(result, console)
    viz.plot(result, save_path="titration.png")
    console.print("[green]Saved titration.png[/green]\n")


def run_kinetics():
    console.rule("[bold blue]Reaction Kinetics (1st order)")
    exp = ChemistryLab.reaction_kinetics(order=1, k=0.15, initial_concentration=2.0)
    result = engine.solve(exp)
    print_result_table(result, console)
    viz.plot(result, save_path="kinetics.png")
    console.print("[green]Saved kinetics.png[/green]\n")


def run_population():
    console.rule("[bold blue]Lotka-Volterra Predator-Prey")
    exp = BiologyLab.population_dynamics(
        prey_initial=40, predator_initial=9,
        alpha=0.1, beta=0.02, delta=0.01, gamma=0.1,
    )
    result = engine.solve(exp)
    print_result_table(result, console)
    viz.plot(result, save_path="predator_prey.png")
    console.print("[green]Saved predator_prey.png[/green]\n")


def run_genetics():
    console.rule("[bold blue]Mendelian Genetics (Dihybrid)")
    exp = BiologyLab.genetics(parent1_genotype="AaBb", parent2_genotype="AaBb")
    result = engine.solve(exp)
    print_result_table(result, console)


def run_full_report():
    console.rule("[bold magenta]Full Lab Report: Projectile")
    exp = PhysicsLab.projectile_motion(velocity=30, angle_deg=50)
    result = engine.solve(exp)
    report = generate_report(result)
    print_report(report, console)


if __name__ == "__main__":
    run_projectile()
    run_pendulum()
    run_circuit()
    run_titration()
    run_kinetics()
    run_population()
    run_genetics()
    run_full_report()
