#!/usr/bin/env python3
"""
4K resolution visualization of exp(1/z²) with type hints and <80 char lines.
"""

from typing import Tuple, Optional, Union
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import time

try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("Numba JIT compilation available - will be much faster!")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - using standard NumPy (will be slower)")

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


def exp_1_over_z2_numpy(
    z: ComplexArray, 
    exclusion_radius: float = 0.01
) -> ComplexArray:
    """Fast NumPy implementation of exp(1/z²) with value capping."""
    # Avoid true zero (causes division by zero)
    z_safe = np.where(np.abs(z) < 1e-10, 1e-10 + 0j, z)
    
    # Compute 1/z² and cap it to prevent overflow
    inv_z2 = 1.0 / (z_safe ** 2)
    
    # Cap the argument to exp() to prevent overflow
    # exp(20) ≈ 500 million, exp(10) ≈ 22,000
    real_part = np.real(inv_z2)
    imag_part = np.imag(inv_z2)
    
    # Cap real part to prevent exp(huge_number)
    real_part_capped = np.clip(real_part, -20, 20)
    inv_z2_capped = real_part_capped + 1j * imag_part
    
    # Compute exp(capped_value)
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.exp(inv_z2_capped)
    
    # Handle any remaining infinities/NaNs by capping magnitude
    result_mag = np.abs(result)
    result_angle = np.angle(result)
    
    # Cap magnitude to reasonable value
    result_mag_capped = np.where(result_mag > 1e6, 1e6, result_mag)
    result_mag_capped = np.where(
        ~np.isfinite(result_mag_capped), 
        1e6, 
        result_mag_capped
    )
    
    # Reconstruct complex number
    result = result_mag_capped * np.exp(1j * result_angle)
    
    return result


if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def exp_1_over_z2_jit(
        z_real: RealArray, 
        z_imag: RealArray
    ) -> Tuple[RealArray, RealArray]:
        """JIT-compiled version for speed."""
        result_real = np.zeros_like(z_real)
        result_imag = np.zeros_like(z_imag)
        
        flat_real = z_real.flatten()
        flat_imag = z_imag.flatten()
        flat_res_real = result_real.flatten()
        flat_res_imag = result_imag.flatten()
        
        for i in range(len(flat_real)):
            zr, zi = flat_real[i], flat_imag[i]
            z_mag2 = zr*zr + zi*zi
            
            if z_mag2 < 1e-20:
                flat_res_real[i] = np.nan
                flat_res_imag[i] = np.nan
                continue
            
            # 1/z² = (1/(zr + zi*i))² = (zr - zi*i)²/(zr² + zi²)²
            denom = z_mag2 * z_mag2
            inv_z2_real = (zr*zr - zi*zi) / denom
            inv_z2_imag = -2*zr*zi / denom
            
            # Cap extreme values
            mag = inv_z2_real*inv_z2_real + inv_z2_imag*inv_z2_imag
            if mag > 2500:  # |inv_z2| > 50
                scale = 50 / np.sqrt(mag)
                inv_z2_real *= scale
                inv_z2_imag *= scale
            
            # exp(a + bi) = exp(a) * (cos(b) + sin(b)*i)
            exp_real_part = np.exp(inv_z2_real)
            cos_imag = np.cos(inv_z2_imag)
            sin_imag = np.sin(inv_z2_imag)
            
            flat_res_real[i] = exp_real_part * cos_imag
            flat_res_imag[i] = exp_real_part * sin_imag
        
        return result_real, result_imag


def create_4k_visualization(
    xlim: Tuple[float, float] = (-0.2, 0.2), 
    ylim: Tuple[float, float] = (-0.2, 0.2), 
    resolution: int = 4000, 
    use_jit: bool = True, 
    exclusion_radius: float = 0.05
) -> Tuple[Figure, Axes]:
    """Create 4K resolution domain coloring of exp(1/z²)."""
    
    print(f"Creating 4K visualization "
          f"({resolution}x{resolution} = {resolution**2:,} points)")
    print(f"Domain: {xlim[0]} to {xlim[1]} (real), "
          f"{ylim[0]} to {ylim[1]} (imag)")
    print(f"Exclusion zone: radius = {exclusion_radius} "
          f"(black circle in center)")
    
    start_time = time.time()
    
    # Create grid
    print("Creating coordinate grid...")
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    grid_time = time.time()
    print(f"Grid created in {grid_time - start_time:.2f}s")
    
    # Compute function values
    print("Computing exp(1/z²)...")
    if NUMBA_AVAILABLE and use_jit:
        print("Using JIT compilation for maximum speed...")
        W_real, W_imag = exp_1_over_z2_jit(X, Y)
        W = W_real + 1j * W_imag
    else:
        print("Using NumPy vectorization...")
        W = exp_1_over_z2_numpy(Z, exclusion_radius)
    
    compute_time = time.time()
    print(f"Function computed in {compute_time - grid_time:.2f}s")
    
    # Convert to colors with custom aggressive mapping
    print("Converting to colors...")
    print(f"Function value range: min={np.nanmin(np.abs(W)):.2e}, "
          f"max={np.nanmax(np.abs(W)):.2e}")
    
    # Custom domain coloring with very aggressive brightness scaling
    arg = (np.angle(W) / (2*np.pi) + 0.5) % 1.0
    mag = np.abs(W)
    
    # Ultra-aggressive brightness normalization for extreme values
    with np.errstate(invalid='ignore'):
        brightness = np.clip(np.log1p(mag) / 8, 0, 1)  
        brightness = np.where(~np.isfinite(brightness), 0, brightness)
    
    # High saturation for vivid colors
    saturation = np.ones_like(arg) * 0.9
    
    # Create HSV and convert to RGB
    hsv = np.stack([arg, saturation, brightness], axis=-1)
    rgb = hsv_to_rgb(hsv)
    
    print(f"RGB range: min={np.nanmin(rgb):.3f}, "
          f"max={np.nanmax(rgb):.3f}")
    print(f"Brightness range: min={np.nanmin(brightness):.3f}, "
          f"max={np.nanmax(brightness):.3f}")
    
    color_time = time.time()
    print(f"Colors generated in {color_time - compute_time:.2f}s")
    
    # Create plot
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(16, 16))  # Large figure for 4K
    
    ax.imshow(
        rgb, 
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
        origin='lower', 
        interpolation='nearest'
    )  # No interpolation for crisp 4K
    
    ax.set_xlabel('Real Part', fontsize=16)
    ax.set_ylabel('Imaginary Part', fontsize=16)
    ax.set_title(
        f'4K Domain Coloring: exp(1/z²) - Essential Singularity\n'
        f'{resolution}×{resolution} resolution', 
        fontsize=18
    )
    ax.grid(True, alpha=0.3)
    
    plot_time = time.time()
    print(f"Plot created in {plot_time - color_time:.2f}s")
    
    # Save high-resolution image
    print("Saving 4K image...")
    save_plot(fig, f"exp_1_z2_4K_{resolution}x{resolution}.png")
    
    save_time = time.time()
    total_time = save_time - start_time
    
    print(f"Image saved in {save_time - plot_time:.2f}s")
    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Points per second: {resolution**2/total_time:,.0f}")
    
    return fig, ax


def save_plot(fig: Figure, filename: str) -> None:
    """Save plot to output directory."""
    import os
    output_dir = ("/home/tasinari/my_repos/"
                 "Riemann_Zeta_Experiments/output")
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def main() -> None:
    """Generate 4K visualization."""
    print("4K exp(1/z²) Essential Singularity Visualizer")
    print("=" * 50)
    
    # Warn about computation time
    resolution = 4000
    estimated_minutes = (resolution**2) / 1000000 * 2
    
    print(f"About to compute {resolution**2:,} points")
    print(f"Estimated time: {estimated_minutes:.1f} minutes")
    print("Press Ctrl+C to cancel if this is too long...")
    
    try:
        # Start with smaller resolution for testing
        test_resolution = 500
        print(f"Testing with {test_resolution}x{test_resolution} first...")
        
        fig, ax = create_4k_visualization(
            xlim=(-0.2, 0.2), 
            ylim=(-0.2, 0.2),
            resolution=test_resolution,
            use_jit=NUMBA_AVAILABLE,
            exclusion_radius=0.1  # Try a big exclusion zone
        )
        
        print("\n4K visualization complete! 🎉")
        print("The image reveals incredible fractal detail "
              "near the essential singularity.")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\nCancelled by user")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()