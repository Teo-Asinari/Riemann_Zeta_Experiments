#!/usr/bin/env python3
"""
Multi-zoom comparison of exp(1/z²) essential singularity.

This script generates domain coloring visualizations of exp(1/z²) 
at different zoom levels to show the fractal-like behavior near z=0.
"""

from typing import List, Dict, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import mpmath
import numpy as np
from domain_coloring_base_typed import (
    visualize_function, 
    ComplexNumber,
    save_plot
)

ZoomLevel = Dict[str, Union[float, str, int]]


def exp_1_over_z_squared(z: ComplexNumber) -> ComplexNumber:
    """Essential singularity: f(z) = exp(1/z²)"""
    if abs(z) < 1e-12:
        return float('nan') + 1j * float('nan')
    return complex(mpmath.exp(1 / (complex(z) ** 2)))


def create_zoom_comparison() -> List[Figure]:
    """Create multiple visualizations at different zoom levels."""
    
    # Define zoom levels (radius around origin)
    zoom_levels: List[ZoomLevel] = [
        {"radius": 1.0, "label": "Wide View", "resolution": 300},
        {"radius": 0.3, "label": "Medium Zoom", "resolution": 350},
        {"radius": 0.1, "label": "Close Zoom", "resolution": 400},
        {"radius": 0.05, "label": "Ultra Zoom", "resolution": 450},
        {"radius": 0.02, "label": "Extreme Zoom", "resolution": 500},
    ]
    
    print("exp(1/z²) Zoom Comparison Generator")
    print("=" * 40)
    
    figures: List[Figure] = []
    
    for i, zoom in enumerate(zoom_levels, 1):
        radius = float(zoom["radius"])
        label = str(zoom["label"])
        resolution = int(zoom["resolution"])
        
        print(f"\n{i}. Generating {label} (radius={radius})...")
        
        # Create visualization
        fig, ax = visualize_function(
            exp_1_over_z_squared,
            f"exp(1/z²) - {label}",
            xlim=(-radius, radius),
            ylim=(-radius, radius),
            resolution=resolution,
            filename=f"exp_1_z2_zoom_{i:02d}_{label.lower().replace(' ', '_')}.png",
            max_mag=20  # Higher max_mag for better contrast
        )
        
        figures.append(fig)
    
    return figures


def create_comparison_grid() -> Figure:
    """Create a single figure with all zoom levels in a grid."""
    print("\nCreating comparison grid...")
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        'exp(1/z²) Essential Singularity - Multi-Zoom Comparison', 
        fontsize=16, y=0.95
    )
    
    zoom_configs: List[ZoomLevel] = [
        {"radius": 1.0, "title": "Wide View\n(±1.0)", "resolution": 200},
        {"radius": 0.3, "title": "Medium Zoom\n(±0.3)", "resolution": 200},
        {"radius": 0.1, "title": "Close Zoom\n(±0.1)", "resolution": 200},
        {"radius": 0.05, "title": "Ultra Zoom\n(±0.05)", "resolution": 200},
        {"radius": 0.02, "title": "Extreme Zoom\n(±0.02)", "resolution": 200},
        {"radius": 0.01, "title": "Maximum Zoom\n(±0.01)", "resolution": 200},
    ]
    
    for idx, config in enumerate(zoom_configs):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        radius = float(config["radius"])
        resolution = int(config["resolution"])
        title = str(config["title"])
        
        print(f"Computing subplot {idx+1}/6 (radius={radius})...")
        
        # Generate data for this zoom level
        from domain_coloring_base_typed import complex_to_hsv
        from matplotlib.colors import hsv_to_rgb
        
        # Create grid
        x = np.linspace(-radius, radius, resolution)
        y = np.linspace(-radius, radius, resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        # Compute function values
        W = np.zeros_like(Z, dtype=complex)
        for i in range(resolution):
            for j in range(resolution):
                W[i, j] = exp_1_over_z_squared(Z[i, j])
        
        # Convert to colors
        hsv = complex_to_hsv(W, max_mag=20)
        rgb = hsv_to_rgb(hsv)
        
        # Plot
        ax.imshow(
            rgb, 
            extent=[-radius, radius, -radius, radius], 
            origin='lower', 
            interpolation='bilinear'
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Real', fontsize=10)
        ax.set_ylabel('Imaginary', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison grid
    save_plot(fig, "exp_1_z2_comparison_grid.png")
    
    return fig


def main() -> None:
    """Generate all zoom comparison visualizations."""
    print("Starting exp(1/z²) zoom comparison generation...")
    
    # Generate individual zoom levels
    individual_figs = create_zoom_comparison()
    
    # Generate comparison grid
    grid_fig = create_comparison_grid()
    
    print("\n" + "="*50)
    print("All visualizations complete!")
    print("Generated files:")
    print("- Individual zoom levels: "
          "exp_1_z2_zoom_01_*.png through exp_1_z2_zoom_05_*.png")
    print("- Comparison grid: exp_1_z2_comparison_grid.png")
    print("\nThese images reveal the fractal-like, self-similar patterns")
    print("that emerge near the essential singularity at z=0!")
    
    # Show the comparison grid
    plt.show()


if __name__ == "__main__":
    main()