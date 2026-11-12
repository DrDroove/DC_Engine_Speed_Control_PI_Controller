"""Reproduce the three PI-controller regimes from Section 1.3.5."""

from pathlib import Path

from pi_controller_graphics import (
    PIParameters,
    classify_mode,
    plot_modes,
    save_individual_plots,
)


# Parameters and Ki values from the document's comparison table.
base = {"a": 20.0, "T": 0.02}
parameter_sets = [
    PIParameters(**base, Kp = 0.01,  Ki=1.8367),  # oscillatory, zeta ~= 0.50
    PIParameters(**base, Kp = 0.09, Ki=4.900),  # aperiodic, zeta = 1.00
    PIParameters(**base, Kp = 0.02, Ki=1.225),  #asymptotic
    PIParameters(**base, Kp = 0.09, Ki=4.761),  # aperiodic, zeta ~= 2.21
    PIParameters(**base, Kp = 0.09, Ki=2.500), #asymptotic
]

modes = [classify_mode(parameters) for parameters in parameter_sets]

print("Classified modes and overcontrol:")
for mode in modes:
    result = mode.overcontrol_details()
    print(
        f"Case {mode.case_number}: {mode.mode_name:12s} "
        f"zeta={mode.zeta:.3f}, "
        f"overcontrol={result.percent:.2f}%, "
        f"peak_time={result.peak_time}, "
        f"setting time={result.settling_time}"
    )

output = Path(__file__).resolve().with_name("pi_graphics_output")
output.mkdir(exist_ok=True)

# One common graph with all three transition responses.
plot_modes(
    modes,
    layout="combined",
    t_end=0.5,
    save_path=output / "all_modes_combined.png",
    title="PI controller — comparison of all three modes",
)

# A single figure containing one subplot per mode.
plot_modes(
    modes,
    layout="separate",
    t_end=[0.5]*5,
    save_path=output / "all_modes_separate_subplots.png",
)

# Three independent PNG files, one per mode.
save_individual_plots(
    modes,
    output / "individual",
    t_ends=[0.5]*5,
)

print(f"Graphs saved under: {output.resolve()}")
