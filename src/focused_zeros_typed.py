#!/usr/bin/env python3
"""
Focused view of the first few Riemann zeta zeros.
"""

from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from zeta_visualizer_typed import create_visualizer, save_visualization


def main() -> None:
    """Focus on the first few zeros."""
    print("Creating focused view of first zeros...")

    visualizer = create_visualizer(precision=50)

    # Focus on the region containing first few zeros
    # First zero is around 0.5 + 14.13i
    fig, ax = visualizer.visualize_magnitude(
        real_range=(0.3, 0.7),  # Narrow around Re(s) = 0.5
        imag_range=(10, 30),  # First few zeros
        resolution=400,  # Higher resolution
    )

    save_visualization(fig, "focused_zeros.png")
    plt.show()


if __name__ == "__main__":
    main()
