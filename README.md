# Riemann Zeta Experiments

**Author:** Teo Asinari  
**Acknowledgment:** Written with [Claude Code](https://claude.ai/code)

## Overview

This project explores the Riemann zeta function ζ(s) through high-precision visualization, with a focus on understanding the distribution of zeros in the complex plane and their connection to the Riemann Hypothesis.

## Features

- High-precision calculation of ζ(s) using `mpmath`
- Visualization of |ζ(s)| magnitude in the complex plane
- Focus on the critical strip (0 < Re(s) < 1) where non-trivial zeros lie
- Identification of both trivial zeros (negative even integers) and non-trivial zeros
- Clean, PEP 8 compliant code with Black formatting

## Key Files

- `zeta_visualizer.py` - Main visualization tool for the Riemann zeta function
- `focused_zeros.py` - High-resolution view of the first few non-trivial zeros
- `requirements.txt` - Python dependencies

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Generate basic zeta function visualization
python zeta_visualizer.py

# Generate focused view of first zeros
python focused_zeros.py
```

## Mathematical Background

The Riemann zeta function is defined as:
- ζ(s) = Σ(1/n^s) for Re(s) > 1
- Extended analytically to the entire complex plane (except s = 1)

### Zeros of Interest

- **Trivial zeros**: s = -2, -4, -6, -8, ... (well understood)
- **Non-trivial zeros**: Located in critical strip 0 < Re(s) < 1
- **Riemann Hypothesis**: All non-trivial zeros have Re(s) = 1/2

The distribution of these zeros is intimately connected to the distribution of prime numbers.

## Visualizations

The project generates visualizations showing:
- The magnitude |ζ(s)| across the complex plane
- Critical line (Re(s) = 1/2) highlighted in red
- Critical strip boundaries marked
- Dark regions indicating zeros where |ζ(s)| ≈ 0

---

*This project represents an exploration into one of mathematics' greatest unsolved problems.*