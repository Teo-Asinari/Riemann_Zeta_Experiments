#!/usr/bin/env python3
"""
Domain coloring visualization for complex functions with essential singularities.

Domain coloring maps complex function values f(z) to colors:
- Hue represents the argument (angle) of f(z)
- Brightness represents the magnitude |f(z)|
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import mpmath


def complex_to_hsv(z, max_mag=10):
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
    func, xlim=(-2, 2), ylim=(-2, 2), resolution=800, title="Domain Coloring"
):
    """
    Create domain coloring plot for a complex function.
    
    Args:
        func: Function that takes complex input and returns complex output
        xlim: Real axis limits
        ylim: Imaginary axis limits
        resolution: Grid resolution
        title: Plot title
    
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

    for i, z in enumerate(flat_z):
        try:
            flat_w[i] = func(z)
        except:
            flat_w[i] = np.nan

        if i % (len(flat_z) // 10) == 0:
            print(f"Progress: {100 * i // len(flat_z)}%")

    W = flat_w.reshape(Z.shape)

    # Convert to colors
    hsv = complex_to_hsv(W)
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


class EssentialSingularityVisualizer:
    """Visualizer for functions with essential singularities."""

    def __init__(self, precision=30):
        """Initialize with specified precision."""
        mpmath.mp.dps = precision
        self.precision = precision

    def exp_over_z(self, z):
        """Essential singularity at z=0: f(z) = exp(1/z)"""
        if abs(z) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.exp(1 / complex(z)))

    def exp_over_z_minus_1(self, z):
        """Essential singularity at z=1: f(z) = exp(1/(z-1))"""
        if abs(z - 1) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.exp(1 / complex(z - 1)))

    def sin_over_z(self, z):
        """Essential singularity at z=0: f(z) = sin(1/z)"""
        if abs(z) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.sin(1 / complex(z)))

    def cos_over_z(self, z):
        """Essential singularity at z=0: f(z) = cos(1/z)"""
        if abs(z) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.cos(1 / complex(z)))

    def exp_z_over_z_squared(self, z):
        """Essential singularity at z=0: f(z) = exp(z)/(z^2)"""
        if abs(z) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.exp(complex(z)) / (complex(z) ** 2))

    def log_z_over_z_minus_1(self, z):
        """Branch point and pole: f(z) = log(z)/(z-1)"""
        if abs(z) < 1e-10 or abs(z - 1) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.log(complex(z)) / (complex(z) - 1))

    def tan_over_z(self, z):
        """Essential singularity at z=0: f(z) = tan(1/z)"""
        if abs(z) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.tan(1 / complex(z)))

    def exp_1_over_z_squared(self, z):
        """Essential singularity at z=0: f(z) = exp(1/z^2)"""
        if abs(z) < 1e-10:
            return np.nan + 1j * np.nan
        return complex(mpmath.exp(1 / (complex(z) ** 2)))

    def visualize_exp_over_z(self, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), resolution=800):
        """Visualize exp(1/z) with essential singularity at origin."""
        return plot_domain_coloring(
            self.exp_over_z,
            xlim,
            ylim,
            resolution,
            "Domain Coloring: exp(1/z) - Essential Singularity at z=0",
        )

    def visualize_sin_over_z(self, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1), resolution=800):
        """Visualize sin(1/z) with essential singularity at origin."""
        return plot_domain_coloring(
            self.sin_over_z,
            xlim,
            ylim,
            resolution,
            "Domain Coloring: sin(1/z) - Essential Singularity at z=0",
        )

    def visualize_cos_over_z(self, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1), resolution=800):
        """Visualize cos(1/z) with essential singularity at origin."""
        return plot_domain_coloring(
            self.cos_over_z,
            xlim,
            ylim,
            resolution,
            "Domain Coloring: cos(1/z) - Essential Singularity at z=0",
        )

    def visualize_tan_over_z(self, xlim=(-0.2, 0.2), ylim=(-0.2, 0.2), resolution=800):
        """Visualize tan(1/z) with essential singularity at origin."""
        return plot_domain_coloring(
            self.tan_over_z,
            xlim,
            ylim,
            resolution,
            "Domain Coloring: tan(1/z) - Essential Singularity at z=0",
        )

    def visualize_exp_1_over_z_squared(
        self, xlim=(-0.08, 0.08), ylim=(-0.08, 0.08), resolution=800
    ):
        """Visualize exp(1/z^2) with essential singularity at origin."""
        return plot_domain_coloring(
            self.exp_1_over_z_squared,
            xlim,
            ylim,
            resolution,
            "Domain Coloring: exp(1/z²) - Essential Singularity at z=0",
        )

    def visualize_exp_z_over_z_squared(
        self, xlim=(-1, 1), ylim=(-1, 1), resolution=600
    ):
        """Visualize exp(z)/z^2 with pole and exponential growth."""
        return plot_domain_coloring(
            self.exp_z_over_z_squared,
            xlim,
            ylim,
            resolution,
            "Domain Coloring: exp(z)/z² - Pole at z=0",
        )


def save_plot(fig, filename):
    """Save plot to output directory."""
    import os

    output_dir = "/home/tasinari/my_repos/Riemann_Zeta_Experiments/output"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")


def main():
    """Demonstrate essential singularity visualizations."""
    print("Essential Singularity Domain Coloring Visualizer")
    print("=" * 50)

    visualizer = EssentialSingularityVisualizer(precision=30)

    # Create zoomed-in visualizations with moderate resolution for speed
    print("\n1. Visualizing exp(1/z) [ZOOMED]...")
    fig1, ax1 = visualizer.visualize_exp_over_z(resolution=300)
    save_plot(fig1, "domain_coloring_exp_over_z_zoomed.png")

    print("\n2. Visualizing sin(1/z) [ZOOMED]...")
    fig2, ax2 = visualizer.visualize_sin_over_z(resolution=300)
    save_plot(fig2, "domain_coloring_sin_over_z_zoomed.png")

    print("\n3. Visualizing cos(1/z) [ZOOMED]...")
    fig3, ax3 = visualizer.visualize_cos_over_z(resolution=300)
    save_plot(fig3, "domain_coloring_cos_over_z_zoomed.png")

    print("\n4. Visualizing exp(1/z²) [ULTRA ZOOMED]...")
    fig4, ax4 = visualizer.visualize_exp_1_over_z_squared(resolution=300)
    save_plot(fig4, "domain_coloring_exp_1_over_z_squared_zoomed.png")

    print("\n5. Visualizing exp(z)/z² [MODERATE ZOOM]...")
    fig5, ax5 = visualizer.visualize_exp_z_over_z_squared(resolution=250)
    save_plot(fig5, "domain_coloring_exp_z_over_z_squared.png")

    print("\nAll visualizations complete!")
    print("Domain coloring legend:")
    print("- Hue (color) = argument of f(z)")
    print("- Brightness = magnitude of f(z)")
    print("- Essential singularities show wild color oscillations")
    print("- Zoomed views reveal incredible detail near singularities")

    plt.show()


if __name__ == "__main__":
    main()
