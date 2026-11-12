import unittest

import numpy as np

from pi_controller_graphics import (
    Case1OscillatoryMode,
    Case2CriticalMode,
    Case3AperiodicMode,
    PIParameters,
    classify_mode,
)


class TestPIControllerGraphics(unittest.TestCase):
    def setUp(self):
        base = {"a": 50.0, "T": 0.125, "Kp": 0.05}
        self.modes = [
            classify_mode(PIParameters(**base, Ki=1.96)),
            classify_mode(PIParameters(**base, Ki=0.49)),
            classify_mode(PIParameters(**base, Ki=0.10)),
        ]

    def test_chapter_135_classification(self):
        self.assertIsInstance(self.modes[0], Case1OscillatoryMode)
        self.assertIsInstance(self.modes[1], Case2CriticalMode)
        self.assertIsInstance(self.modes[2], Case3AperiodicMode)

    def test_all_responses_start_at_zero_and_converge_to_one(self):
        for mode in self.modes:
            self.assertAlmostEqual(mode.H(0.0), 0.0, places=12)
            self.assertAlmostEqual(mode.H(100.0), 1.0, places=8)

    def test_overcontrol_is_nonnegative_and_matches_expected_example(self):
        values = [mode.check_for_overcontrol() for mode in self.modes]
        self.assertAlmostEqual(values[0], 22.637, places=2)
        self.assertAlmostEqual(values[1], 1.529, places=2)
        self.assertEqual(values[2], 0.0)
        self.assertTrue(all(value >= 0.0 for value in values))

    def test_analytic_peak_matches_dense_numeric_search(self):
        for mode in self.modes:
            t = np.linspace(0.0, 4.0, 300_000)
            numeric_peak = float(np.max(mode.H(t)))
            analytic_peak = mode.overcontrol_details().peak_value
            # For a response with no overshoot, the global supremum is 1
            # and is reached only as t tends to infinity.
            sampled_supremum = max(1.0, numeric_peak)
            self.assertAlmostEqual(analytic_peak, sampled_supremum, delta=2e-4)

    def test_scalar_and_array_calls(self):
        mode = self.modes[0]
        self.assertIsInstance(mode.H(0.1), float)
        values = mode.H([0.0, 0.1, 0.2])
        self.assertEqual(values.shape, (3,))


if __name__ == "__main__":
    unittest.main()
