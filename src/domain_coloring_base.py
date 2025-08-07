#!/usr/bin/env python3
"""
Modular domain coloring visualization for complex functions.

This module provides the core domain coloring functionality that can be used
with arbitrary complex functions passed as arguments.
"""

from typing import Tuple, Callable, Union, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import mpmath
import os

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
ComplexNumber = Union[complex, np.complex128]
ComplexFunction = Callable[[ComplexNumber], ComplexNumber]


def complex_to_hsv(z: ComplexArray, max_mag: float = 10) -> RealArray:
    """
    Convert complex numbers to HSV color values for domain coloring.
    
    Args:
        z: Complex array
        max_mag: Maximum magnitude for brightness scaling
    
    Returns:
        HSV array with shape (..., 3)
    """
    # Handle NaN and infinite values first
    z = np.where(np.isfinite(z), z, 0)

    # Calculate argument (hue) - maps to [0, 1]
    with np.errstate(invalid="ignore"):
        arg = np.angle(z) / (2 * np.pi) + 0.5

    # Calculate magnitude with overflow protection
    with np.errstate(over="ignore", invalid="ignore"):
        mag = np.abs(z)
        # Cap extremely large values
        mag = np.where(mag > 1e10, 1e10, mag)
        mag = np.where(~np.isfinite(mag), 1e10, mag)

    # Logarithmic scaling for better visualization
    brightness = np.tanh(np.log1p(mag) / np.log1p(max_mag))

    # Saturation (keep high for vivid colors)
    saturation = np.ones_like(arg) * 0.8

    # Stack into HSV format
    hsv = np.stack([arg, saturation, brightness], axis=-1)
    return hsv


def plot_domain_coloring(
    func: ComplexFunction,
    xlim: Tuple[float, float] = (-2, 2),
    ylim: Tuple[float, float] = (-2, 2),
    resolution: int = 500,
    title: str = "Domain Coloring",
    max_mag: float = 10,
) -> Tuple[Figure, Axes]:
    """
    Create domain coloring plot for a complex function.
    
    Args:
        func: Function that takes complex input and returns complex output
        xlim: Real axis limits (tuple)
        ylim: Imaginary axis limits (tuple)
        resolution: Grid resolution (int)
        title: Plot title (string)
        max_mag: Maximum magnitude for color scaling
    
    Returns:
        Figure and axis objects
    """
    # Create complex grid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    print(f"Computing {title} over {resolution}x{resolution} grid...")

    # Evaluate function
    W = np.zeros_like(Z, dtype=complex)
    flat_z = Z.flatten()
    flat_w = W.flatten()

    total_points = len(flat_z)
    for i, z in enumerate(flat_z):
        try:
            flat_w[i] = func(z)
        except Exception:
            flat_w[i] = np.nan + 1j * np.nan

        if i % (total_points // 10) == 0:
            print(f"Progress: {100 * i // total_points}%")

    W = flat_w.reshape(Z.shape)

    # Convert to colors
    hsv = complex_to_hsv(W, max_mag=max_mag)
    rgb = hsv_to_rgb(hsv)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        rgb,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        interpolation="bilinear",
    )

    ax.set_xlabel("Real Part", fontsize=12)
    ax.set_ylabel("Imaginary Part", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig, ax


def create_function_from_string(func_str: str) -> ComplexFunction:
    """
    Create a function from a string representation.
    
    Args:
        func_str: String like "exp(1/z)" or "sin(1/z)"
    
    Returns:
        Function that can be called with complex arguments
    """

    def func(z: ComplexNumber) -> ComplexNumber:
        if abs(z) < 1e-10:
            return np.nan + 1j * np.nan

        try:
            # Replace 'z' with the actual value and evaluate
            expr = func_str.replace("z", f"({complex(z)})")
            # Use mpmath for evaluation
            result = eval(f"mpmath.{expr}")
            return complex(result)
        except Exception:
            return np.nan + 1j * np.nan

    return func


def save_plot(fig: Figure, filename: str, output_dir: Optional[str] = None) -> None:
    """Save plot to output directory."""
    if output_dir is None:
        output_dir = "/home/tasinari/my_repos/" "Riemann_Zeta_Experiments/output"

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")


def visualize_function(
    func: Union[ComplexFunction, str],
    func_name: str,
    xlim: Tuple[float, float] = (-1, 1),
    ylim: Tuple[float, float] = (-1, 1),
    resolution: int = 400,
    filename: Optional[str] = None,
    max_mag: float = 10,
) -> Tuple[Figure, Axes]:
    """
    Visualize a complex function using domain coloring.
    
    Args:
        func: Complex function or string representation
        func_name: Name for the function (used in title)
        xlim: Real axis limits
        ylim: Imaginary axis limits
        resolution: Grid resolution
        filename: Output filename (auto-generated if None)
        max_mag: Maximum magnitude for color scaling
    
    Returns:
        Figure and axis objects
    """
    # Convert string to function if needed
    if isinstance(func, str):
        func = create_function_from_string(func)

    # Generate title
    title = f"Domain Coloring: {func_name}"

    # Create visualization
    fig, ax = plot_domain_coloring(func, xlim, ylim, resolution, title, max_mag)

    # Save if filename provided
    if filename:
        save_plot(fig, filename)

    return fig, ax


def main() -> None:
    """Example usage."""

    def exp_1_over_z_squared(z: ComplexNumber) -> ComplexNumber:
        if abs(z) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.exp(1 / (complex(z) ** 2)))

    fig, ax = visualize_function(
        exp_1_over_z_squared,
        "exp(1/z²)",
        xlim=(-0.5, 0.5),
        ylim=(-0.5, 0.5),
        resolution=300,
        filename="test_modular_domain_coloring.png",
    )

    plt.show()


if __name__ == "__main__":
    main()
