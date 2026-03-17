"""Matplotlib-based visualization for simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from labsim.models import SimResult


class ResultVisualizer:
    """Generate publication-quality plots from SimResult objects."""

    STYLE_DEFAULTS = {
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
        "lines.linewidth": 2,
    }

    def __init__(self):
        plt.rcParams.update(self.STYLE_DEFAULTS)

    def plot(
        self,
        result: SimResult,
        save_path: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Auto-detect the best plot type for *result* and render it."""
        if not result.y or (isinstance(result.y, list) and len(result.y) == 0):
            return self._plot_empty(result)

        name = result.experiment_name.lower()

        if "projectile" in name:
            fig = self._plot_trajectory(result)
        elif "pendulum" in name or "lotka" in name:
            fig = self._plot_multi_timeseries(result)
        elif "titration" in name:
            fig = self._plot_titration(result)
        else:
            fig = self._plot_timeseries(result)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Plot types
    # ------------------------------------------------------------------

    def _plot_trajectory(self, result: SimResult) -> plt.Figure:
        """X-Y trajectory (projectile motion)."""
        fig, ax = plt.subplots()
        x, y = result.y[0], result.y[1]
        ax.plot(x, y, color="royalblue")
        ax.fill_between(x, 0, y, alpha=0.1, color="royalblue")
        ax.set_xlabel(result.labels.get("x", "x"))
        ax.set_ylabel(result.labels.get("y", "y"))
        ax.set_title(result.labels.get("title", result.experiment_name))
        ax.set_ylim(bottom=0)
        self._annotate_summary(ax, result)
        fig.tight_layout()
        return fig

    def _plot_timeseries(self, result: SimResult) -> plt.Figure:
        """Single variable vs time."""
        fig, ax = plt.subplots()
        y = result.y[0] if isinstance(result.y, list) else result.y
        ax.plot(result.t, y, color="darkorange")
        ax.set_xlabel(result.labels.get("x", "Time"))
        ax.set_ylabel(result.labels.get("y", "Value"))
        ax.set_title(result.labels.get("title", result.experiment_name))
        self._annotate_summary(ax, result)
        fig.tight_layout()
        return fig

    def _plot_multi_timeseries(self, result: SimResult) -> plt.Figure:
        """Multiple state variables vs time (pendulum, predator-prey)."""
        n_vars = len(result.y)
        fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4 * n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]
        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
        for i, (ax, yi) in enumerate(zip(axes, result.y)):
            label = result.labels.get(f"y{i}", f"State {i}")
            ax.plot(result.t, yi, color=colors[i % len(colors)], label=label)
            ax.set_ylabel(label)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel(result.labels.get("x", "Time"))
        fig.suptitle(
            result.labels.get("title", result.experiment_name),
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig

    def _plot_titration(self, result: SimResult) -> plt.Figure:
        """pH vs volume added, with equivalence point marker."""
        fig, ax = plt.subplots()
        ph = result.y[0] if isinstance(result.y, list) else result.y
        ax.plot(result.t, ph, color="purple", linewidth=2)
        ax.set_xlabel(result.labels.get("x", "Volume (mL)"))
        ax.set_ylabel(result.labels.get("y", "pH"))
        ax.set_title(result.labels.get("title", "Titration Curve"))

        # Mark equivalence point
        eq_vol = result.metadata.get("equivalence_volume_mL")
        if eq_vol is not None:
            idx = int(np.argmin(np.abs(result.t - eq_vol)))
            ax.axvline(eq_vol, color="red", linestyle="--", alpha=0.7, label="Equivalence")
            ax.plot(eq_vol, ph[idx], "ro", markersize=8)
            ax.legend()

        # Mark pKa / half-equivalence
        pka = result.metadata.get("pka")
        if pka is not None and eq_vol is not None:
            ax.axhline(pka, color="green", linestyle=":", alpha=0.5, label=f"pKa = {pka:.2f}")
            ax.legend()

        fig.tight_layout()
        return fig

    def _plot_empty(self, result: SimResult) -> plt.Figure:
        """Placeholder for algebraic (non-ODE) experiments."""
        fig, ax = plt.subplots()
        ax.text(
            0.5, 0.5,
            f"{result.experiment_name}\n(algebraic result - see report)",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=14,
        )
        ax.set_title(result.labels.get("title", result.experiment_name))
        ax.axis("off")
        fig.tight_layout()
        return fig

    @staticmethod
    def _annotate_summary(ax, result: SimResult) -> None:
        """Add summary stats as text box on the plot."""
        if not result.summary:
            return
        lines = [f"{k}: {v:.4g}" for k, v in result.summary.items()]
        text = "\n".join(lines)
        props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7)
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )
