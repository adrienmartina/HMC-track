import tempfile
import unittest
from pathlib import Path

import numpy as np

from matrix_hmc_track.metastability_analysis import (
    autocorrelation_fft,
    integrated_autocorrelation_time,
    load_R2_all,
    load_escape_stats,
)


class TestMetastabilityAnalysis(unittest.TestCase):
    def test_ar1_integrated_autocorrelation_time(self) -> None:
        rng = np.random.default_rng(1234)
        phi = 0.8
        n = 30000
        noise = rng.normal(scale=np.sqrt(1.0 - phi**2), size=n)
        series = np.empty(n, dtype=np.float64)
        series[0] = noise[0]
        for i in range(1, n):
            series[i] = phi * series[i - 1] + noise[i]

        result = integrated_autocorrelation_time(series, method="sokal", c=5)
        expected_tau = 0.5 + phi / (1.0 - phi)

        self.assertEqual(result["n"], n)
        self.assertGreater(result["window"], 0)
        self.assertAlmostEqual(result["acf"][0], 1.0)
        self.assertLess(abs(result["tau_int"] - expected_tau), 1.0)
        self.assertNotIn("len < 50", result["warnings"])

    def test_autocorrelation_fft_constant_series(self) -> None:
        acf = autocorrelation_fft(np.ones(5))
        np.testing.assert_allclose(acf, np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]))

    def test_load_escape_stats_first_escape_and_censor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            np.savez(
                run_dir / "escape.npz",
                iteration=np.asarray([10, 20, 30]),
                escaped=np.asarray([False, True, False]),
                converged=np.asarray([True, True, True]),
                reference_trx2=np.asarray(2.0),
                tolerance_atol=np.asarray(0.1),
                tolerance_rtol=np.asarray(0.05),
            )
            stats = load_escape_stats(run_dir)

            self.assertEqual(stats["first_escape_iteration"], 20)
            self.assertTrue(stats["escaped_bool"])
            self.assertFalse(stats["censored_bool"])
            self.assertEqual(stats["convergence_fraction"], 1.0)
            self.assertEqual(stats["reference_trx2"], 2.0)
            self.assertEqual(stats["tolerance"], 0.2)
            np.testing.assert_array_equal(
                stats["raw"]["iteration"], np.asarray([10, 20, 30])
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            escape_path = Path(tmpdir) / "escape.npz"
            np.savez(
                escape_path,
                iteration=np.asarray([1, 2, 3]),
                escaped=np.asarray([False, True, False]),
                converged=np.asarray([True, False, True]),
            )
            stats = load_escape_stats(escape_path)

            self.assertEqual(stats["first_escape_iteration"], 2)
            self.assertFalse(stats["first_escape_converged"])
            self.assertIsNone(stats["first_reliable_escape_iteration"])
            self.assertTrue(stats["escaped_bool"])
            self.assertFalse(stats["censored_bool"])
            self.assertEqual(stats["first_nonconverged_escape_iteration"], 2)
            self.assertEqual(stats["nonconverged_count"], 1)
            self.assertIn(
                "escaped classifications include non-converged descents",
                stats["warnings"],
            )

    def test_load_R2_all_from_tiny_synthetic_evals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            evals = np.asarray(
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0],
                        [100.0, 100.0],
                    ],
                    [
                        [1.0, -1.0],
                        [2.0, -2.0],
                        [3.0, -3.0],
                        [4.0, -4.0],
                        [100.0, 100.0],
                    ],
                ],
                dtype=np.float64,
            )
            np.savez(run_dir / "evals.npz", values=evals)

            expected = np.asarray(
                [
                    np.sum(evals[0, :4, :] ** 2),
                    np.sum(evals[1, :4, :] ** 2),
                ]
            )
            np.testing.assert_allclose(load_R2_all(run_dir), expected)

    def test_load_R2_all_accepts_common_alias_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            evals = np.ones((2, 4, 3), dtype=np.float64)
            np.savez(run_dir / "evals.npz", eigenvalues=evals)

            np.testing.assert_allclose(load_R2_all(run_dir), np.asarray([12.0, 12.0]))


if __name__ == "__main__":
    unittest.main()
