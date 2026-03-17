"""Click CLI for LABSIM."""

from __future__ import annotations

import json

import click
from rich.console import Console

from labsim.labs.physics import PhysicsLab
from labsim.labs.chemistry import ChemistryLab
from labsim.labs.biology import BiologyLab
from labsim.simulator.engine import SimulationEngine
from labsim.simulator.parameters import ExperimentParameters
from labsim.simulator.visualizer import ResultVisualizer
from labsim.report import generate_report, print_report, print_result_table

console = Console()

# Map (lab, experiment) -> factory callable
_FACTORIES = {
    ("physics", "projectile"): PhysicsLab.projectile_motion,
    ("physics", "pendulum"): PhysicsLab.pendulum,
    ("physics", "circuit"): PhysicsLab.ohms_law_circuit,
    ("physics", "lens"): PhysicsLab.thin_lens,
    ("chemistry", "ideal_gas"): ChemistryLab.ideal_gas,
    ("chemistry", "kinetics"): ChemistryLab.reaction_kinetics,
    ("chemistry", "titration"): ChemistryLab.titration,
    ("chemistry", "molecular"): ChemistryLab.molecular_geometry,
    ("biology", "cell_division"): BiologyLab.cell_division,
    ("biology", "genetics"): BiologyLab.genetics,
    ("biology", "population"): BiologyLab.population_dynamics,
}


def _resolve_experiment(lab: str, experiment: str, params: dict):
    """Look up the factory and build an Experiment object."""
    key = (lab.lower(), experiment.lower().replace("-", "_").replace(" ", "_"))
    factory = _FACTORIES.get(key)
    if factory is None:
        available = [f"{k[0]}/{k[1]}" for k in _FACTORIES]
        raise click.ClickException(
            f"Unknown experiment '{lab}/{experiment}'. "
            f"Available: {', '.join(available)}"
        )
    return factory(**params)


def _parse_params(param_strings: tuple[str, ...]) -> dict:
    """Parse 'key=value' CLI arguments into a dict with type coercion."""
    result = {}
    for item in param_strings:
        if "=" not in item:
            raise click.BadParameter(f"Expected key=value, got '{item}'")
        k, v = item.split("=", 1)
        # Try numeric coercion
        try:
            v_parsed = int(v)
        except ValueError:
            try:
                v_parsed = float(v)
            except ValueError:
                v_parsed = v
        result[k] = v_parsed
    return result


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

@click.group()
@click.version_option(package_name="labsim")
def cli():
    """LABSIM - Virtual STEM Lab simulator."""
    pass


# ------------------------------------------------------------------
# labsim run
# ------------------------------------------------------------------

@cli.command()
@click.argument("experiment_name")
@click.option("--lab", "-l", default="physics", help="Lab discipline")
@click.option("--plot", "-p", default=None, help="Save plot to file")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.argument("params", nargs=-1)
def run(experiment_name: str, lab: str, plot: str, json_output: bool, params: tuple):
    """Run a simulation experiment.

    Examples:

        labsim run projectile angle_deg=60 velocity=25

        labsim run pendulum --lab physics length=2.0

        labsim run kinetics --lab chemistry order=2 k=0.05
    """
    parsed = _parse_params(params)
    exp = _resolve_experiment(lab, experiment_name, parsed)

    engine = SimulationEngine()
    result = engine.solve(exp)

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        return

    print_result_table(result, console)

    if plot:
        viz = ResultVisualizer()
        viz.plot(result, save_path=plot)
        console.print(f"\n[green]Plot saved to {plot}[/green]")


# ------------------------------------------------------------------
# labsim experiment
# ------------------------------------------------------------------

@cli.command()
@click.argument("lab")
@click.argument("experiment_name")
@click.option("--plot", "-p", default=None, help="Save plot to file")
@click.argument("params", nargs=-1)
def experiment(lab: str, experiment_name: str, plot: str, params: tuple):
    """Run an experiment with lab/name syntax.

    Examples:

        labsim experiment physics projectile angle_deg=45

        labsim experiment chemistry titration ka=1.0e-5

        labsim experiment biology population alpha=0.2
    """
    parsed = _parse_params(params)
    ep = ExperimentParameters(lab=lab, experiment=experiment_name, params=parsed)
    merged = ep.get_merged_params()

    exp = _resolve_experiment(lab, experiment_name, merged)

    engine = SimulationEngine()
    result = engine.solve(exp)

    print_result_table(result, console)

    if plot:
        viz = ResultVisualizer()
        viz.plot(result, save_path=plot)
        console.print(f"\n[green]Plot saved to {plot}[/green]")


# ------------------------------------------------------------------
# labsim report
# ------------------------------------------------------------------

@cli.command()
@click.argument("experiment_name")
@click.option("--lab", "-l", default="physics", help="Lab discipline")
@click.option("--format", "fmt", default="rich", type=click.Choice(["rich", "json"]))
@click.argument("params", nargs=-1)
def report(experiment_name: str, lab: str, fmt: str, params: tuple):
    """Generate a detailed lab report.

    Examples:

        labsim report projectile --lab physics

        labsim report titration --lab chemistry --format json
    """
    parsed = _parse_params(params)
    ep = ExperimentParameters(lab=lab, experiment=experiment_name, params=parsed)
    merged = ep.get_merged_params()

    exp = _resolve_experiment(lab, experiment_name, merged)

    engine = SimulationEngine()
    result = engine.solve(exp)
    lab_report = generate_report(result)

    if fmt == "json":
        click.echo(lab_report.model_dump_json(indent=2))
    else:
        print_report(lab_report, console)


# ------------------------------------------------------------------
# labsim list
# ------------------------------------------------------------------

@cli.command("list")
@click.option("--lab", "-l", default=None, help="Filter by lab discipline")
def list_experiments(lab: str | None):
    """List available experiments."""
    experiments = ExperimentParameters.list_experiments(lab)
    for discipline, names in experiments.items():
        console.print(f"\n[bold cyan]{discipline.title()} Lab[/bold cyan]")
        for name in names:
            console.print(f"  - {name}")


if __name__ == "__main__":
    cli()
