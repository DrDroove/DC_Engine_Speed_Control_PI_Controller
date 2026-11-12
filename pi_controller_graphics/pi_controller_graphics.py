"""Step-response graphics for the PI-controlled motor described in Chapter 1.

The model follows Sections 1.1 and 1.3.1--1.3.5 of the supplied document.
Only NumPy is needed for calculations; Matplotlib is imported when plotting.

Usage
-----
    from pi_controller_graphics import PIParameters, classify_mode, plot_modes

    params = PIParameters(a=50, T=0.125, Kp=0.05, Ki=1.96)
    mode = classify_mode(params)

    values = mode.H([0.0, 0.1, 0.2])
    overshoot_percent = mode.check_for_overcontrol()
    fig, ax = mode.plot(t_end=0.8)

    # Compare all three regimes on one set of axes or as separate subplots:
    modes = [classify_mode(p) for p in parameter_sets]
    fig, axes = plot_modes(modes, layout="combined")
    fig, axes = plot_modes(modes, layout="separate")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class PIParameters:
    """Physical and controller parameters of the closed-loop PI system.

    The document assumes a > 0, T > 0, Kp > 0 and Ki > 0.  Kp=0 is
    accepted as a useful limiting case; Ki must remain strictly positive.
    """

    a: float
    T: float
    Kp: float
    Ki: float

    def __post_init__(self) -> None:
        for name in ("a", "T", "Kp", "Ki"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number; got {value!r}.")
            object.__setattr__(self, name, value)

        if self.a <= 0:
            raise ValueError("a must be > 0.")
        if self.T <= 0:
            raise ValueError("T must be > 0.")
        if self.Kp < 0:
            raise ValueError("Kp must be >= 0.")
        if self.Ki <= 0:
            raise ValueError("Ki must be > 0.")

    @property
    def alpha(self) -> float:
        """alpha = (1 + a*Kp) / (2*T)."""
        return (1.0 + self.a * self.Kp) / (2.0 * self.T)

    @property
    def omega_n(self) -> float:
        """Undamped natural frequency omega_n = sqrt(a*Ki/T)."""
        return math.sqrt(self.a * self.Ki / self.T)

    @property
    def zeta(self) -> float:
        """Damping ratio zeta = alpha / omega_n."""
        return self.alpha / self.omega_n

    @property
    def omega_d(self) -> float:
        """Damped natural frequency; defined only in the oscillatory regime."""
        value = self.omega_n**2 - self.alpha**2
        return math.sqrt(max(0.0, value))

    @property
    def beta(self) -> float:
        """Positive root-separation parameter for the aperiodic regime."""
        value = self.alpha**2 - self.omega_n**2
        return math.sqrt(max(0.0, value))


@dataclass(frozen=True)
class OvercontrolResult:
    """Details of the maximum overshoot above the final value H(infinity)=1.

    ``percent`` is the usual percentage overshoot.  It is exactly 0.0 when
    the response does not rise above 1.  ``ratio`` is the dimensionless
    quantity Mp = H_peak - 1 used in the document.
    """

    percent: float
    ratio: float
    peak_value: float
    peak_time: float | None
    settling_time: float

    @property
    def present(self) -> bool:
        return self.percent > 0.0


class VirtualMode(ABC):
    """Common interface implemented by each of the three root regimes."""

    mode_name: str = "Undefined mode"
    case_number: int = 0

    def __init__(self, parameters: PIParameters) -> None:
        self.parameters = parameters

    @property
    def a(self) -> float:
        return self.parameters.a

    @property
    def T(self) -> float:
        return self.parameters.T

    @property
    def Kp(self) -> float:
        return self.parameters.Kp

    @property
    def Ki(self) -> float:
        return self.parameters.Ki

    @property
    def alpha(self) -> float:
        return self.parameters.alpha

    @property
    def omega_n(self) -> float:
        return self.parameters.omega_n

    @property
    def zeta(self) -> float:
        return self.parameters.zeta

    @abstractmethod
    def H(self, t: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Return the unit-step transition response H(t)."""

    @abstractmethod
    def _candidate_peak(self) -> tuple[float | None, float]:
        """Return (time, H(time)) for an overshoot peak, or (None, 1)."""

    def __call__(self, t: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Allow a mode instance to be used directly as H(t)."""
        return self.H(t)

    def overcontrol_details(self, epsilon: float = 0.05) -> OvercontrolResult:
        """Return peak value/time and overshoot in ratio and percent."""
        peak_time, peak_value = self._candidate_peak()
        ratio = max(0.0, float(peak_value) - 1.0)
        # Remove tiny floating-point artefacts around the final value.
        if ratio < 1e-12:
            ratio = 0.0
            peak_value = 1.0
            peak_time = None
        return OvercontrolResult(
            percent=100.0 * ratio,
            ratio=ratio,
            peak_value=float(peak_value),
            peak_time=peak_time,
            settling_time=self.calculate_settling_time(epsilon),
        )

    def calculate_settling_time(
        self,
        epsilon: float = 0.05
    ) -> float:
        """
        Calculate the time after which H(t) permanently stays
        inside the band [1-epsilon, 1+epsilon].

        Default epsilon=0.05 corresponds to a 5% working band.

        Returns
        -------
        float
            Settling time in seconds.
        """

        if not math.isfinite(epsilon) or not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be between 0 and 1.")

        # Find a time horizon beyond which the error envelope
        # is already smaller than epsilon.
        t_end = self.default_time_end(
            epsilon=epsilon,
            padding=1.05
        )

        # Adapt sampling density to the fastest characteristic
        # time scale of the corresponding mode.
        if isinstance(self, Case1OscillatoryMode):
            characteristic_rate = max(self.alpha, self.omega_d)

        elif isinstance(self, Case2CriticalMode):
            characteristic_rate = self.alpha

        else:
            characteristic_rate = max(
                abs(self.p1),
                abs(self.p2)
            )

        n_points = max(
            5001,
            math.ceil(t_end * characteristic_rate * 300) + 1
        )

        # Calculate the response over the time interval.
        time = np.linspace(0.0, t_end, n_points)

        response = np.asarray(self.H(time), dtype=float)

        # Find all points outside the working band.
        deviation = np.abs(response - 1.0)

        outside = np.flatnonzero(deviation > epsilon)

        # If the response never leaves the working band.
        if outside.size == 0:
            return 0.0

        # The last sampled point outside the band.
        last_index = int(outside[-1])

        if last_index >= len(time) - 1:
            raise RuntimeError(
                "Increase the calculation time horizon."
            )

        # The permanent-entry crossing lies between these points.
        lo = float(time[last_index])
        hi = float(time[last_index + 1])

        # Refine the crossing using bisection.
        for _ in range(80):

            mid = (lo + hi) / 2.0

            error = abs(float(self.H(mid)) - 1.0)

            if error > epsilon:
                lo = mid
            else:
                hi = mid

        return hi

    def check_for_overcontrol(self) -> float:
        """Return percentage overshoot; return 0.0 if there is no overshoot."""
        return self.overcontrol_details().percent

    # Convenient spelling for callers who prefer the shorter method name.
    check_overcontrol = check_for_overcontrol

    def default_time_end(self, epsilon: float = 0.01, padding: float = 1.15) -> float:
        """Estimate a useful plot horizon from the exponential decay envelope.

        The estimate targets an error envelope of ``epsilon`` and then adds
        ``padding`` so the settling tail remains visible.  ``plot(t_end=...)``
        can always be used to override it.
        """
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be between 0 and 1.")
        if padding <= 0:
            raise ValueError("padding must be > 0.")

        k = 1.0 / self.T - self.alpha
        if isinstance(self, Case1OscillatoryMode):
            amplitude = math.hypot(1.0, k / self.parameters.omega_d)
            horizon = math.log(max(amplitude / epsilon, 1.0)) / self.alpha
        elif isinstance(self, Case2CriticalMode):
            # Bound: exp(-alpha*t) * (1 + abs(k)*t).  For Kp >= 0,
            # abs(k) <= alpha, so this bound is non-increasing.
            abs_k = abs(k)

            def bound(time_value: float) -> float:
                return math.exp(-self.alpha * time_value) * (1.0 + abs_k * time_value)

            lo, hi = 0.0, max(1.0 / self.alpha, 1e-9)
            while bound(hi) > epsilon:
                hi *= 2.0
                if hi > 1e12:
                    raise RuntimeError("Could not determine a finite plot horizon.")
            for _ in range(80):
                mid = (lo + hi) / 2.0
                if bound(mid) > epsilon:
                    lo = mid
                else:
                    hi = mid
            horizon = hi
        else:
            assert isinstance(self, Case3AperiodicMode)
            amplitude = abs(self.B) + abs(self.C)
            if amplitude <= epsilon:
                horizon = 1.0 / abs(self.p1)
            else:
                horizon = max(0.0, math.log(amplitude / epsilon) / abs(self.p1))

        return max(padding * horizon, 1e-3)

    def plot(
        self,
        *,
        t_end: float | None = None,
        n_points: int = 1200,
        show_tolerance: bool = False,
        epsilon: float = 0.01,
        save_path: str | Path | None = None,
        show: bool = False,
        title: str | None = None,
    ):
        """Plot this mode and return ``(figure, axes)``."""
        return plot_modes(
            [self],
            layout="combined",
            t_end=t_end,
            n_points=n_points,
            show_tolerance=show_tolerance,
            epsilon=epsilon,
            save_path=save_path,
            show=show,
            title=title or f"Case {self.case_number}: {self.mode_name}",
        )

    # Alias matching the wording commonly used in scripts/notebooks.
    print_graph = plot


class Case1OscillatoryMode(VirtualMode):
    """Case 1: 0 < zeta < 1, complex-conjugate characteristic roots."""

    mode_name = "Oscillatory"
    case_number = 1

    @property
    def omega_d(self) -> float:
        return self.parameters.omega_d

    @property
    def k(self) -> float:
        # The coefficient multiplying sin(omega_d*t) in Eq. (12).
        return 1.0 / self.T - self.alpha

    def H(self, t: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        time = _validated_time(t)
        wd = self.omega_d
        values = 1.0 - np.exp(-self.alpha * time) * (
            np.cos(wd * time) + (self.k / wd) * np.sin(wd * time)
        )
        return _scalar_or_array(t, values)

    def _candidate_peak(self) -> tuple[float | None, float]:
        wd = self.omega_d
        # H'(t)=0 iff c*cos(wd*t) + d*sin(wd*t) = 0.
        # Since H'(0)=a*Kp/T >= 0, the first positive stationary point
        # is the first local maximum (the later maxima have a smaller
        # exponential envelope).
        c = self.a * self.Kp / self.T
        d = (self.alpha * self.k + wd**2) / wd
        theta = math.atan2(-c, d) % math.pi
        if theta < 1e-12:
            theta = math.pi
        peak_time = theta / wd
        peak_value = float(self.H(peak_time))
        if peak_value <= 1.0 + 1e-12:
            return None, 1.0
        return peak_time, peak_value


class Case2CriticalMode(VirtualMode):
    """Case 2: zeta = 1, repeated real characteristic roots."""

    mode_name = "Critical"
    case_number = 2

    @property
    def k(self) -> float:
        return 1.0 / self.T - self.alpha

    def H(self, t: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        time = _validated_time(t)
        values = 1.0 - np.exp(-self.alpha * time) * (1.0 + self.k * time)
        return _scalar_or_array(t, values)

    def _candidate_peak(self) -> tuple[float | None, float]:
        # H'(t)=exp(-alpha*t) * (alpha*(1+k*t)-k).
        # The only possible stationary point is t=1/alpha - 1/k.
        if abs(self.k) < 1e-15:
            return None, 1.0
        peak_time = 1.0 / self.alpha - 1.0 / self.k
        if peak_time <= 0.0:
            return None, 1.0
        peak_value = float(self.H(peak_time))
        if peak_value <= 1.0 + 1e-12:
            return None, 1.0
        return peak_time, peak_value


class Case3AperiodicMode(VirtualMode):
    """Case 3: zeta > 1, two distinct negative real roots."""

    mode_name = "Aperiodic"
    case_number = 3

    def __init__(self, parameters: PIParameters) -> None:
        super().__init__(parameters)
        self.beta = self.parameters.beta
        self.p1 = -self.alpha + self.beta  # slower (less negative) root
        self.p2 = -self.alpha - self.beta  # faster (more negative) root
        if not (self.p1 < 0.0 and self.p2 < 0.0):
            raise ValueError("Aperiodic mode requires two negative real roots.")

        # H(t) = 1 + B*exp(p1*t) + C*exp(p2*t).
        # This is algebraically equivalent to Eq. (16) in the document.
        denominator = self.p1 - self.p2
        self.B = -(self.p1 + 1.0 / self.T) / denominator
        self.C = -(self.p2 + 1.0 / self.T) / (self.p2 - self.p1)

    def H(self, t: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        time = _validated_time(t)
        values = 1.0 + self.B * np.exp(self.p1 * time) + self.C * np.exp(self.p2 * time)
        return _scalar_or_array(t, values)

    def _candidate_peak(self) -> tuple[float | None, float]:
        bp1 = self.B * self.p1
        cp2 = self.C * self.p2
        if abs(bp1) < 1e-15:
            return None, 1.0

        ratio = -cp2 / bp1
        # A positive-time extremum exists only when ratio > 1.
        if ratio <= 1.0 or not math.isfinite(ratio):
            return None, 1.0

        peak_time = math.log(ratio) / (self.p1 - self.p2)
        if peak_time <= 0.0:
            return None, 1.0
        peak_value = float(self.H(peak_time))
        if peak_value <= 1.0 + 1e-12:
            return None, 1.0
        return peak_time, peak_value


def classify_mode(
    parameters: PIParameters | None = None,
    *,
    a: float | None = None,
    T: float | None = None,
    Kp: float | None = None,
    Ki: float | None = None,
    critical_tolerance: float = 1e-9,
) -> VirtualMode:
    """Classify the PI-system mode and return its corresponding mode object.

    Pass either a ``PIParameters`` instance or all four scalar parameters.
    ``critical_tolerance`` controls how close zeta must be to 1 to be treated
    as the critical case (using both relative and absolute tolerance).
    """
    if critical_tolerance < 0:
        raise ValueError("critical_tolerance must be >= 0.")

    scalar_values = (a, T, Kp, Ki)
    if parameters is None:
        if any(value is None for value in scalar_values):
            raise TypeError("Pass PIParameters or all of a, T, Kp and Ki.")
        parameters = PIParameters(a=a, T=T, Kp=Kp, Ki=Ki)  # type: ignore[arg-type]
    elif any(value is not None for value in scalar_values):
        raise TypeError("Pass either PIParameters or scalar parameters, not both.")

    zeta = parameters.zeta
    if math.isclose(zeta, 1.0, rel_tol=critical_tolerance, abs_tol=critical_tolerance):
        return Case2CriticalMode(parameters)
    if zeta < 1.0:
        return Case1OscillatoryMode(parameters)
    return Case3AperiodicMode(parameters)


def plot_modes(
    modes: Sequence[VirtualMode],
    *,
    layout: Literal["combined", "separate"] = "combined",
    t_end: float | Sequence[float] | None = None,
    n_points: int = 1200,
    show_tolerance: bool = False,
    epsilon: float = 0.01,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot multiple modes overlaid (``combined``) or in separate subplots.

    Parameters
    ----------
    modes:
        Mode objects, usually obtained from ``classify_mode``.
    layout:
        ``"combined"`` overlays all responses on one axes;
        ``"separate"`` creates one vertically stacked subplot per response.
    t_end:
        Optional common end time, or one end time per mode.  If omitted,
        each mode's decay envelope determines a suitable horizon.
    show_tolerance:
        Draw the 1 +/- epsilon settling corridor when True.
    save_path:
        Optional file path to save the resulting figure.
    show:
        Call ``matplotlib.pyplot.show()`` when True.

    Returns
    -------
    (figure, axes)
        ``axes`` is a single Axes for combined layout, or an array of Axes
        for separate layout.
    """
    if not modes:
        raise ValueError("modes must contain at least one mode object.")
    if layout not in ("combined", "separate"):
        raise ValueError("layout must be 'combined' or 'separate'.")
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be between 0 and 1.")

    import matplotlib.pyplot as plt

    ends = _resolve_time_ends(modes, t_end)
    if layout == "combined":
        if t_end is None:
            common_end = max(ends)
            ends = [common_end] * len(modes)
        fig, ax = plt.subplots(figsize=figsize or (9.0, 5.2))
        for mode, end in zip(modes, ends):
            time = np.linspace(0.0, end, n_points)
            label = f"{mode.mode_name} (ζ={mode.zeta:.3g}, Ki={mode.Ki:g})"
            ax.plot(time, mode.H(time), label=label)
        _decorate_axes(ax, show_tolerance=show_tolerance, epsilon=epsilon)
        ax.set_xlabel("t, s")
        ax.set_ylabel("H(t)")
        ax.set_title(title or "PI controller: transition responses")
        ax.legend()
        axes = ax
    else:
        if len(modes) == 1:
            rows, cols = 1, 1
        else:
            rows, cols = len(modes), 1
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=figsize or (9.0, max(3.0, 2.8 * len(modes))),
            squeeze=False,
        )
        axes = axes[:, 0]
        for index, (mode, end) in enumerate(zip(modes, ends)):
            ax = axes[index]
            time = np.linspace(0.0, end, n_points)
            ax.plot(time, mode.H(time), label=f"{mode.mode_name} (ζ={mode.zeta:.3g})")
            _decorate_axes(ax, show_tolerance=show_tolerance, epsilon=epsilon)
            ax.set_xlim(0.0, end)
            ax.set_xlabel("t, s")
            ax.set_ylabel("H(t)")
            ax.set_title(f"Case {mode.case_number}: {mode.mode_name}")
            ax.legend(loc="best")
        if title:
            fig.suptitle(title)
        fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def save_individual_plots(
    modes: Sequence[VirtualMode],
    directory: str | Path,
    *,
    t_ends: Sequence[float] | None = None,
    show_tolerance: bool = False,
    epsilon: float = 0.01,
    n_points: int = 1200,
    file_prefix: str = "pi_mode",
) -> list[Path]:
    """Save one PNG per mode and return the created file paths."""
    if not modes:
        raise ValueError("modes must contain at least one mode object.")
    if t_ends is not None and len(t_ends) != len(modes):
        raise ValueError("t_ends must have exactly one value per mode.")

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, mode in enumerate(modes):
        end = None if t_ends is None else t_ends[index]
        path = output_dir / f"{file_prefix}_{mode.case_number}_{mode.mode_name.lower()}.png"
        mode.plot(
            t_end=end,
            n_points=n_points,
            show_tolerance=show_tolerance,
            epsilon=epsilon,
            save_path=path,
            show=False,
        )
        paths.append(path)
    return paths


def _validated_time(t: float | Sequence[float] | np.ndarray) -> np.ndarray:
    time = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(time)):
        raise ValueError("t must contain only finite values.")
    if np.any(time < 0.0):
        raise ValueError("The step response is defined here for t >= 0.")
    return time


def _scalar_or_array(original_t, values: np.ndarray) -> float | np.ndarray:
    if np.ndim(original_t) == 0:
        return float(np.asarray(values))
    return np.asarray(values, dtype=float)


def _resolve_time_ends(
    modes: Sequence[VirtualMode], t_end: float | Sequence[float] | None
) -> list[float]:
    if t_end is None:
        return [mode.default_time_end() for mode in modes]
    if np.ndim(t_end) == 0:
        value = float(t_end)  # type: ignore[arg-type]
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("t_end must be a finite number > 0.")
        return [value] * len(modes)
    values = [float(value) for value in t_end]  # type: ignore[union-attr]
    if len(values) != len(modes):
        raise ValueError("When t_end is a sequence, give one value per mode.")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Every t_end value must be finite and > 0.")
    return values


def _decorate_axes(ax, *, show_tolerance: bool, epsilon: float) -> None:
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Final value H(∞)=1")
    if show_tolerance:
        ax.axhspan(
            1.0 - epsilon,
            1.0 + epsilon,
            alpha=0.12,
            label=f"±{epsilon:.1%} tolerance",
        )
    ax.set_ylim(bottom=min(-0.02, ax.get_ylim()[0]))
    ax.grid(True, alpha=0.3)


__all__ = [
    "PIParameters",
    "OvercontrolResult",
    "VirtualMode",
    "Case1OscillatoryMode",
    "Case2CriticalMode",
    "Case3AperiodicMode",
    "classify_mode",
    "plot_modes",
    "save_individual_plots",
]
