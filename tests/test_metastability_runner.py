import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import matrix_hmc_track as hmc
from matrix_hmc_track.algebra import random_hermitian
from matrix_hmc_track.models.pikkt4d_type3 import PIKKTTypeIIModel as PIKKTTypeIIIModel
from matrix_hmc_track.simulation import _classical_gradient_descent


class TestType3TrivialVacuum(unittest.TestCase):
    def setUp(self):
        hmc.configure(device="cpu", precision="complex128")

    def test_spin_zero_initializes_exact_zero_for_n2(self):
        model = PIKKTTypeIIIModel(
            ncol=2,
            couplings=[0.1, 1.0],
            bosonic=True,
            spin=0,
        )
        model.load_fresh()
        torch.testing.assert_close(
            model.get_state(),
            torch.zeros_like(model.get_state()),
            rtol=0,
            atol=0,
        )

    def test_bosonic_mode_drops_fermion_determinant(self):
        X = random_hermitian(2, batchsize=4)
        bosonic_model = PIKKTTypeIIIModel(ncol=2, couplings=[0.2, 1.0], bosonic=True)
        full_model = PIKKTTypeIIIModel(ncol=2, couplings=[0.2, 1.0], bosonic=False)

        torch.testing.assert_close(
            bosonic_model.potential(X),
            bosonic_model.bosonic_potential(X),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            full_model.potential(X),
            full_model.bosonic_potential(X) + full_model.ferm_potential(X),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertNotAlmostEqual(
            float(full_model.potential(X).real),
            float(bosonic_model.potential(X).real),
            places=8,
        )

    def test_descent_from_zero_converges_to_zero_reference(self):
        model = PIKKTTypeIIIModel(
            ncol=2,
            couplings=[0.1, 1.0],
            bosonic=True,
            spin=0,
        )
        model.load_fresh()
        grad = model.force(model.get_state())
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=1e-12)

        _, info = _classical_gradient_descent(
            model,
            model.get_state(),
            max_steps=20,
            step_size=0.01,
            grad_tol=1e-12,
            min_step_size=1e-14,
            max_backtracks=10,
        )
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(float(info["trx2"]), 0.0, places=14)


class TestMetastabilityCliSmoke(unittest.TestCase):
    def test_cli_spin0_bosonic_escape_run_writes_stats(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "main.py",
                "--model",
                "pikkt4d_type3",
                "--ncol",
                "2",
                "--coupling",
                "0.2",
                "1.0",
                "--bosonic",
                "--spin",
                "0",
                "--fresh",
                "--niters",
                "2",
                "--step-size",
                "0.02",
                "--nsteps",
                "4",
                "--track-escape",
                "--no-save-step-eigenvalues",
                "--precision",
                "complex128",
                "--seed",
                "123",
                "--name",
                "cli_smoke",
                "--data-path",
                tmp,
                "--no-save",
            ]
            result = subprocess.run(
                cmd,
                cwd=root,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

            run_dirs = sorted(Path(tmp).glob("cli_smoke_*"))
            self.assertEqual(len(run_dirs), 1, result.stdout + result.stderr)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "evals.npz").is_file())
            self.assertTrue((run_dir / "escape.npz").is_file())
            self.assertFalse((run_dir / "step_evals.npz").exists())

            with open(run_dir / "run_stats.json", "r", encoding="utf-8") as f:
                stats = json.load(f)
            self.assertEqual(stats["completed_trajectories"], 2)
            self.assertTrue(stats["bosonic"])
            self.assertEqual(stats["spin"], 0.0)
            self.assertTrue(stats["step_eigenvalues_disabled"])
            self.assertTrue(stats["track_escape"])
            self.assertEqual(stats["nsteps"], 4)
            self.assertAlmostEqual(stats["dt"], 0.005)


if __name__ == "__main__":
    unittest.main()
