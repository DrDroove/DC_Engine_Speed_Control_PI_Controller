# PI-controller graphics

A small Python module for calculating and plotting the unit-step response
`H(t)` of the PI-controlled motor described in the supplied document,
Chapter 1, especially Sections 1.3.1–1.3.5.

## Install

```bash
python -m pip install -r requirements_pi_graphics.txt
```

## Basic use

```python
from pi_controller_graphics import PIParameters, classify_mode

parameters = PIParameters(a=50, T=0.125, Kp=0.05, Ki=1.96)
mode = classify_mode(parameters)

print(mode.mode_name)                  # Oscillatory
print(mode.H(0.2))                      # H(t) at t=0.2 s
print(mode.check_for_overcontrol())     # percentage; 0.0 means none
print(mode.overcontrol_details())      # percentage, ratio, peak and peak time

fig, ax = mode.plot(t_end=0.8, show=True)
# Same as mode.plot(...):
# fig, ax = mode.print_graph(t_end=0.8, show=True)
```

You can also classify directly from scalar parameters:

```python
mode = classify_mode(a=50, T=0.125, Kp=0.05, Ki=1.96)
```

## Compare several parameter sets

```python
from pi_controller_graphics import PIParameters, classify_mode, plot_modes

base = {"a": 50, "T": 0.125, "Kp": 0.05}
ki_values = [1.96, 0.49, 0.10]
modes = [classify_mode(PIParameters(**base, Ki=ki)) for ki in ki_values]

# All responses on the same axes:
plot_modes(modes, layout="combined", t_end=3.0, show=True)

# One subplot per response:
plot_modes(modes, layout="separate", t_end=[0.8, 1.2, 3.0], show=True)
```

For three separate image files, use:

```python
from pi_controller_graphics import save_individual_plots

paths = save_individual_plots(
    modes,
    "graphics/individual",
    t_ends=[0.8, 1.2, 3.0],
)
print(paths)
```

Run `python example_pi_graphics.py` to reproduce the comparison data from
the table in Section 1.3.5 and save combined, subplot, and individual PNGs.

## What the classifier does

The classifier computes

- `alpha = (1 + a*Kp) / (2*T)`
- `omega_n = sqrt(a*Ki/T)`
- `zeta = alpha / omega_n`

and returns one of:

- `Case1OscillatoryMode` for `zeta < 1`;
- `Case2CriticalMode` for `zeta` approximately equal to `1`;
- `Case3AperiodicMode` for `zeta > 1`.

The equality tolerance is configurable through `classify_mode(...,
critical_tolerance=1e-9)`. The base class provides the common interface;
`H(t)` and the mode-specific peak calculation are overridden by each case.

## Overcontrol convention and source note

`check_for_overcontrol()` returns percentage overshoot, i.e.
`100 * max(0, H_peak - 1)`. It returns the numeric value `0.0` when there is
no overcontrol. `overcontrol_details()` additionally returns the peak time,
peak value, and dimensionless overshoot ratio `Mp = H_peak - 1`.

There is an algebraic inconsistency in the supplied document's critical-mode
paragraph. Its stated response is

`H(t) = 1 - exp(-alpha*t) * (1 + k*t)`, where `k = 1/T - alpha`.

Differentiating this expression gives the stationary point

`t_peak = 1/alpha - 1/k` (when it is positive).

For the Section 1.3.5 values `a=50`, `T=0.125`, `Kp=0.05`, and `Ki=0.49`,
this gives a small overshoot of approximately **1.53%**. The document's later
statement that the critical case has no overshoot for `Kp >= 0` does not follow
from that response formula. This module follows the displayed `H(t)` formula
and its derivative for the numerical check, rather than hard-coding the
contradictory prose conclusion.

The module uses NumPy for calculations and Matplotlib for plots. No SciPy
installation is required.
