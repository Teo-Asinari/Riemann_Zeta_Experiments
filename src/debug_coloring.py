#!/usr/bin/env python3
"""
Debug the domain coloring for exp(1/z²)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def debug_exp_1_z2():
    """Debug what's happening with the coloring"""

    # Small test grid
    x = np.linspace(-0.1, 0.1, 100)
    y = np.linspace(-0.1, 0.1, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Compute function avoiding the center
    mask = np.abs(Z) < 0.01  # Exclude very center
    Z_safe = np.where(mask, 0.01 + 0j, Z)

    # Simple computation
    W = np.exp(1 / (Z_safe ** 2))
    W = np.where(mask, 0 + 0j, W)  # Set center to 0

    print(f"Z range: {np.min(np.abs(Z)):.3f} to {np.max(np.abs(Z)):.3f}")
    print(f"W range: {np.min(np.abs(W)):.2e} to {np.max(np.abs(W)):.2e}")
    print(f"W finite values: {np.sum(np.isfinite(W))} / {W.size}")

    # Test different color mappings
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Just magnitude
    axes[0, 0].imshow(
        np.log1p(np.abs(W)),
        extent=[-0.1, 0.1, -0.1, 0.1],
        cmap="viridis",
        origin="lower",
    )
    axes[0, 0].set_title("log(1 + |W|)")

    # 2. Just argument
    axes[0, 1].imshow(
        np.angle(W), extent=[-0.1, 0.1, -0.1, 0.1], cmap="hsv", origin="lower"
    )
    axes[0, 1].set_title("arg(W)")

    # 3. Simple HSV mapping
    arg = (np.angle(W) / (2 * np.pi) + 0.5) % 1.0
    mag = np.abs(W)
    mag_norm = np.tanh(np.log1p(mag) / 10)  # Very aggressive normalization

    hsv = np.stack([arg, np.ones_like(arg) * 0.8, mag_norm], axis=-1)
    rgb = hsv_to_rgb(hsv)

    axes[1, 0].imshow(rgb, extent=[-0.1, 0.1, -0.1, 0.1], origin="lower")
    axes[1, 0].set_title("Domain coloring (aggressive)")

    # 4. Even more aggressive
    mag_norm2 = np.clip(np.log1p(mag) / 5, 0, 1)
    hsv2 = np.stack([arg, np.ones_like(arg) * 0.9, mag_norm2], axis=-1)
    rgb2 = hsv_to_rgb(hsv2)

    axes[1, 1].imshow(rgb2, extent=[-0.1, 0.1, -0.1, 0.1], origin="lower")
    axes[1, 1].set_title("Domain coloring (ultra aggressive)")

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    debug_exp_1_z2()
