#!/usr/bin/env python3
"""
Riemann Zeta Function Visualizer

This module provides visualization tools for the Riemann zeta function ζ(s)
in the complex plane, with particular focus on the critical strip.
"""

from typing import Tuple, Union, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import mpmath

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
ComplexNumber = Union[complex, np.complex128]


class RiemannZetaVisualizer:
    """Visualizer for the Riemann zeta function in the complex plane."""

    def __init__(self, precision: int = 50) -> None:
        """Initialize with specified precision for mpmath calculations."""
        mpmath.mp.dps = precision
        self.precision = precision

    def _compute_single_zeta(self, s: ComplexNumber) -> ComplexNumber:
        """Calculate ζ(s) for a single complex number."""
        try:
            return complex(mpmath.zeta(complex(s)))
        except Exception:
            return np.nan + 1j * np.nan

    def _compute_zeta_array(self, s_array: ComplexArray) -> ComplexArray:
        """Calculate ζ(s) for an array of complex numbers."""
        result = np.zeros_like(s_array, dtype=complex)
        flat_s = s_array.flatten()
        flat_result = result.flatten()

        for i, s in enumerate(flat_s):
            flat_result[i] = self._compute_single_zeta(s)

        return result

    def zeta_function(
        self, s: Union[ComplexNumber, ComplexArray]
    ) -> Union[ComplexNumber, ComplexArray]:
        """Calculate ζ(s) using mpmath for high precision."""
        if np.isscalar(s):
            return self._compute_single_zeta(s)
        else:
            return self._compute_zeta_array(s)

    def create_complex_grid(
        self,
        real_range: Tuple[float, float],
        imag_range: Tuple[float, float],
        resolution: int = 500,
    ) -> Tuple[RealArray, RealArray, ComplexArray]:
        """Create a grid of complex numbers for visualization."""
        real_vals = np.linspace(real_range[0], real_range[1], resolution)
        imag_vals = np.linspace(imag_range[0], imag_range[1], resolution)
        real_grid, imag_grid = np.meshgrid(real_vals, imag_vals)
        complex_grid = real_grid + 1j * imag_grid
        return real_grid, imag_grid, complex_grid

    def _compute_zeta_on_grid(
        self, complex_grid: ComplexArray, resolution: int
    ) -> ComplexArray:
        """Compute zeta function values on a complex grid."""
        zeta_values = np.zeros_like(complex_grid, dtype=complex)
        total_points = resolution * resolution

        for i in range(resolution):
            for j in range(resolution):
                s = complex_grid[i, j]
                zeta_values[i, j] = self._compute_single_zeta(s)

            if i % 50 == 0:
                current = i * resolution
                print(f"Progress: {current}/{total_points} points computed")

        return zeta_values

    def _prepare_magnitude_data(self, zeta_values: ComplexArray) -> RealArray:
        """Process zeta values to prepare magnitude for visualization."""
        magnitude = np.abs(zeta_values)
        # Cap extreme values for better visualization
        magnitude = np.where(magnitude > 100, 100, magnitude)
        magnitude = np.where(magnitude < 0.01, 0.01, magnitude)
        return magnitude

    def _create_base_plot(
        self,
        magnitude: RealArray,
        real_range: Tuple[float, float],
        imag_range: Tuple[float, float],
    ) -> Tuple[Figure, Axes, plt.QuadMesh]:
        """Create the base matplotlib plot with magnitude data."""
        fig, ax = plt.subplots(figsize=(12, 10))

        extent = [real_range[0], real_range[1], imag_range[0], imag_range[1]]

        im = ax.imshow(
            magnitude,
            extent=extent,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            norm=LogNorm(vmin=0.01, vmax=100),
        )

        return fig, ax, im

    def _add_critical_lines(self, ax: Axes) -> None:
        """Add critical line and strip boundaries to the plot."""
        # Critical line at Re(s) = 1/2
        ax.axvline(
            x=0.5,
            color="red",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
            label="Critical Line (Re(s) = 1/2)",
        )

        # Critical strip boundaries
        ax.axvline(
            x=0,
            color="orange",
            linestyle=":",
            alpha=0.6,
            label="Critical Strip Boundary",
        )
        ax.axvline(x=1, color="orange", linestyle=":", alpha=0.6)

    def _finalize_plot(
        self, fig: Figure, ax: Axes, im: plt.QuadMesh
    ) -> Tuple[Figure, Axes]:
        """Add labels, legend, and colorbar to finalize the plot."""
        ax.set_xlabel("Real Part", fontsize=12)
        ax.set_ylabel("Imaginary Part", fontsize=12)
        ax.set_title("Magnitude of Riemann Zeta Function |ζ(s)|", fontsize=14)
        ax.legend()

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("|ζ(s)|", fontsize=12)

        plt.tight_layout()
        return fig, ax

    def visualize_magnitude(
        self,
        real_range: Tuple[float, float] = (-2, 4),
        imag_range: Tuple[float, float] = (-20, 20),
        resolution: int = 300,
    ) -> Tuple[Figure, Axes]:
        """Visualize the magnitude |ζ(s)| in the complex plane."""
        print("Computing ζ(s) magnitude over grid...")

        # Create grid
        _, _, complex_grid = self.create_complex_grid(
            real_range, imag_range, resolution
        )

        # Compute zeta values
        zeta_values = self._compute_zeta_on_grid(complex_grid, resolution)

        # Prepare magnitude data
        magnitude = self._prepare_magnitude_data(zeta_values)

        # Create plot
        fig, ax, im = self._create_base_plot(magnitude, real_range, imag_range)

        # Add critical lines
        self._add_critical_lines(ax)

        # Finalize plot
        return self._finalize_plot(fig, ax, im)

    def visualize_zeros_region(
        self, imag_range: Tuple[float, float] = (0, 50), resolution: int = 800
    ) -> Tuple[Figure, Axes]:
        """High-resolution view of the critical strip."""
        return self.visualize_magnitude(
            real_range=(0, 1), imag_range=imag_range, resolution=resolution
        )


def create_visualizer(precision: int = 30) -> RiemannZetaVisualizer:
    """Factory function to create a RiemannZetaVisualizer."""
    return RiemannZetaVisualizer(precision=precision)


def save_visualization(fig: Figure, filename: str = "zeta_magnitude.png") -> None:
    """Save the visualization to a file."""
    full_path = "/home/tasinari/my_repos/" "Riemann_Zeta_Experiments/output/" + filename
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved as '{filename}'")


def main() -> None:
    """Main function to demonstrate the visualizer."""
    print("Riemann Zeta Function Visualizer")
    print("=" * 40)

    # Create visualizer
    visualizer = create_visualizer(precision=30)

    # Create visualization
    print("Creating magnitude visualization...")
    fig, ax = visualizer.visualize_magnitude(resolution=200)

    # Save and show
    save_visualization(fig)
    plt.show()


if __name__ == "__main__":
    main()
