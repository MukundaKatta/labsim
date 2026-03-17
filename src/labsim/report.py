"""Rich-formatted lab report generation."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from labsim.models import LabReport, SimResult


def generate_report(result: SimResult) -> LabReport:
    """Build a LabReport from a SimResult."""
    # Build conclusion from summary
    summary_lines = [f"  {k}: {v:.4g}" for k, v in result.summary.items()]
    conclusion = (
        "Simulation completed successfully.\n" + "\n".join(summary_lines)
        if summary_lines
        else "Simulation completed. See metadata for algebraic results."
    )

    # Extract analytic results if present
    analytic = result.metadata.get("analytic_result", {})
    extra_params = {
        k: v
        for k, v in analytic.items()
        if isinstance(v, (int, float, str, bool))
    }

    return LabReport(
        title=result.experiment_name,
        discipline=result.discipline,
        objective=f"Simulate and analyse: {result.experiment_name}",
        theory=_theory_for(result.experiment_name),
        parameters={**extra_params},
        results_summary=result.summary,
        conclusion=conclusion,
    )


def print_report(report: LabReport, console: Console | None = None) -> None:
    """Render a LabReport to the terminal using Rich."""
    if console is None:
        console = Console()

    # Header
    header = Text(f"Lab Report: {report.title}", style="bold cyan")
    console.print(Panel(header, box=box.DOUBLE, expand=False))

    # Info table
    info = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    info.add_column("Field", style="bold")
    info.add_column("Value")
    info.add_row("Discipline", report.discipline.value.title())
    info.add_row("Date", report.date.strftime("%Y-%m-%d %H:%M:%S"))
    info.add_row("Objective", report.objective)
    console.print(info)

    # Theory
    if report.theory:
        console.print(Panel(report.theory, title="Theory", border_style="blue"))

    # Parameters
    if report.parameters:
        ptable = Table(title="Parameters", box=box.ROUNDED)
        ptable.add_column("Parameter", style="green")
        ptable.add_column("Value", style="yellow")
        for k, v in report.parameters.items():
            ptable.add_row(str(k), str(v))
        console.print(ptable)

    # Results
    if report.results_summary:
        rtable = Table(title="Results Summary", box=box.ROUNDED)
        rtable.add_column("Metric", style="magenta")
        rtable.add_column("Value", style="bold white")
        for k, v in report.results_summary.items():
            rtable.add_row(k, f"{v:.6g}")
        console.print(rtable)

    # Conclusion
    console.print(
        Panel(report.conclusion, title="Conclusion", border_style="green")
    )


def print_result_table(result: SimResult, console: Console | None = None) -> None:
    """Quick summary table for CLI output."""
    if console is None:
        console = Console()

    console.print(f"\n[bold cyan]{result.experiment_name}[/bold cyan]")
    console.print(f"[dim]{result.discipline.value.title()} Lab[/dim]\n")

    if result.summary:
        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Metric", style="green")
        table.add_column("Value", style="bold")
        for k, v in result.summary.items():
            table.add_row(k, f"{v:.6g}")
        console.print(table)

    # Show algebraic results from metadata
    analytic = result.metadata.get("analytic_result", {})
    if analytic:
        table = Table(title="Analytic Results", box=box.ROUNDED)
        table.add_column("Quantity", style="cyan")
        table.add_column("Value", style="bold yellow")
        for k, v in analytic.items():
            table.add_row(k, str(v))
        console.print(table)


# ------------------------------------------------------------------
# Theory blurbs
# ------------------------------------------------------------------

def _theory_for(name: str) -> str:
    theories = {
        "Projectile Motion": (
            "A projectile follows a parabolic trajectory under uniform gravity.\n"
            "  x(t) = v0*cos(theta)*t\n"
            "  y(t) = v0*sin(theta)*t - (1/2)*g*t^2\n"
            "With quadratic drag: F_drag = -0.5*Cd*|v|*v"
        ),
        "Simple Pendulum": (
            "The non-linear pendulum ODE:\n"
            "  d^2(theta)/dt^2 = -(g/L)*sin(theta) - b*(d(theta)/dt)\n"
            "Small-angle period: T = 2*pi*sqrt(L/g)"
        ),
        "Ohm's Law Circuit": (
            "Ohm's Law: V = I*R\n"
            "Series: R_total = R1 + R2 + ...\n"
            "Parallel: 1/R_total = 1/R1 + 1/R2 + ..."
        ),
        "Thin Lens Optics": (
            "Thin lens equation: 1/f = 1/do + 1/di\n"
            "Magnification: M = -di/do\n"
            "Image height: h_i = M * h_o"
        ),
        "Ideal Gas Law": (
            "PV = nRT\n"
            "Isothermal process: P = nRT/V\n"
            "Work (isothermal): W = nRT*ln(V2/V1)"
        ),
        "Acid-Base Titration": (
            "Henderson-Hasselbalch: pH = pKa + log([A-]/[HA])\n"
            "At equivalence: moles acid = moles base\n"
            "After equivalence: pH dominated by excess OH-"
        ),
        "Cell Division (Logistic Growth)": (
            "Logistic growth: dN/dt = r*N*(1 - N/K)\n"
            "Solution: N(t) = K / (1 + ((K-N0)/N0)*exp(-r*t))\n"
            "Doubling time (exponential phase): t_d = ln(2)/r"
        ),
        "Lotka-Volterra Predator-Prey": (
            "Prey:     dx/dt = alpha*x - beta*x*y\n"
            "Predator: dy/dt = delta*x*y - gamma*y\n"
            "Produces characteristic oscillations in both populations."
        ),
    }
    # Fuzzy match
    for key, text in theories.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return text
    return ""
