"""Dry-run smoke tests for every built-in model.

Params are taken from representative runs in batch.sh.  dry_run=True skips
actual HMC trajectories so these tests exercise model construction,
load_fresh(), path generation, and metadata writing — not sampling.
"""

import importlib
import tempfile
import unittest
from pathlib import Path

import matrix_hmc_track as hmc
from matrix_hmc_track.models.pikkt10d import PIKKT10DModel
from matrix_hmc_track.models.pikkt4d_type1 import PIKKTTypeIModel
from matrix_hmc_track.models.pikkt4d_type2 import PIKKTTypeIIModel
from matrix_hmc_track.models.yangmills import YangMillsModel
from matrix_hmc_track.models.adjoint_det import AdjointDetModel
from matrix_hmc_track.models.susyym_3d import SUSYYM3DModel

_mm = importlib.import_module("matrix_hmc_track.models.1mm")
OneMatrixPolynomialModel = _mm.OneMatrixPolynomialModel


def _run(model, *, step_size, nsteps, niters=100, **kwargs):
    """Helper: dry-run into a fresh temp output dir, return the model."""
    with tempfile.TemporaryDirectory() as tmp:
        result = hmc.run(
            model,
            step_size=step_size,
            nsteps=nsteps,
            niters=niters,
            output=tmp,
            name="test",
            dry_run=True,
            **kwargs,
        )
    return result


class TestPIKKT10D(unittest.TestCase):
    """pikkt10d — from active batch.sh run (ncol=20, g=0.1, spin=0)."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex64")

    def test_dry_run_trivial_vacuum(self):
        # matrix-hmc-track --model pikkt10d --ncol 20 --coupling 0.1 --step-size 0.07 --nsteps 150 --spin 0 --pfaffian-every 1
        model = PIKKT10DModel(ncol=20, couplings=[0.1], spin=0, pfaffian_every=1)
        self.assertIs(_run(model, step_size=0.07, nsteps=150, niters=500), model)

    def test_dry_run_stronger_coupling(self):
        # matrix-hmc-track --model pikkt10d --ncol 20 --coupling 1 --step-size 0.07 --nsteps 150 --spin 0
        model = PIKKT10DModel(ncol=20, couplings=[1.0], spin=0, pfaffian_every=1)
        self.assertIs(_run(model, step_size=0.07, nsteps=150, niters=500), model)

    def test_dry_run_massless(self):
        # matrix-hmc-track --model pikkt10d --massless --ncol 20 --coupling 1 --step-size 0.3 --nsteps 300
        model = PIKKT10DModel(ncol=20, couplings=[1.0], massless=True)
        self.assertIs(_run(model, step_size=0.3, nsteps=300, niters=500), model)


class TestPIKKT4DTypeI(unittest.TestCase):
    """pikkt4d_type1 — from batch.sh scan runs."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex64")

    def test_dry_run_standard(self):
        # matrix-hmc-track --model pikkt4d_type1 --ncol 50 --coupling 150 --step-size 1.5 --nsteps 250
        model = PIKKTTypeIModel(ncol=50, couplings=[150.0])
        self.assertIs(_run(model, step_size=1.5, nsteps=250, niters=600), model)

    def test_dry_run_eta_deformation(self):
        # matrix-hmc-track --model pikkt4d_type1 --ncol 50 --coupling 150 --eta 0.5 --step-size 1.5 --nsteps 250
        model = PIKKTTypeIModel(ncol=50, couplings=[150.0], eta=0.5)
        self.assertIs(_run(model, step_size=1.5, nsteps=250, niters=400), model)

    def test_dry_run_massless(self):
        # matrix-hmc-track --model pikkt4d_type1 --massless --ncol 30 --coupling 1.0 --step-size 1.0 --nsteps 250
        model = PIKKTTypeIModel(ncol=30, couplings=[1.0], massless=True)
        self.assertIs(_run(model, step_size=1.0, nsteps=250, niters=1500), model)


class TestPIKKT4DTypeII(unittest.TestCase):
    """pikkt4d_type2 — from batch.sh trivialScanRun entries."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex64")

    def test_dry_run_trivial_spin0(self):
        # matrix-hmc-track --model pikkt4d_type2 --ncol 30 --coupling 0.1 1 --step-size 0.3 --nsteps 500 --spin 0
        model = PIKKTTypeIIModel(ncol=30, couplings=[0.1, 1.0], spin=0)
        self.assertIs(_run(model, step_size=0.3, nsteps=500, niters=400), model)

    def test_dry_run_large_coupling_spin0(self):
        # matrix-hmc-track --model pikkt4d_type2 --ncol 48 --coupling 100 1 --step-size 1 --nsteps 300 --spin 0
        model = PIKKTTypeIIModel(ncol=48, couplings=[100.0, 1.0], spin=0)
        self.assertIs(_run(model, step_size=1.0, nsteps=300, niters=600), model)

    def test_dry_run_spin1(self):
        # matrix-hmc-track --model pikkt4d_type2 --ncol 48 --coupling 10 1 --step-size 1 --nsteps 200 --spin 1
        model = PIKKTTypeIIModel(ncol=48, couplings=[10.0, 1.0], spin=1)
        self.assertIs(_run(model, step_size=1.0, nsteps=200, niters=1000), model)

    def test_dry_run_bosonic(self):
        model = PIKKTTypeIIModel(ncol=20, couplings=[1.0, 1.0], bosonic=True)
        self.assertIs(_run(model, step_size=0.3, nsteps=100), model)

    def test_run_directory_includes_hmc_parameters(self):
        model = PIKKTTypeIIModel(ncol=54, couplings=[0.0059, 1.0])
        paths = model.build_paths(
            "FindCrit",
            "outputs",
            step_size=0.01,
            nsteps=500,
            niters=300,
        )
        self.assertTrue(
            paths["dir"].endswith(
                "FindCrit_pikkt4d_type2_g0.0059_omega1.0_N54_dt0.01_steps500_iters300"
            )
        )

    def test_bosonic_run_directory_prefixes_model_name(self):
        model = PIKKTTypeIIModel(ncol=54, couplings=[0.0059, 1.0], bosonic=True)
        paths = model.build_paths(
            "FindCrit",
            "outputs",
            step_size=0.01,
            nsteps=500,
            niters=300,
        )
        self.assertTrue(
            paths["dir"].endswith(
                "FindCrit_bospikkt4d_type2_g0.0059_omega1.0_N54_dt0.01_steps500_iters300"
            )
        )

    def test_hmc_run_uses_parameterized_directory_name(self):
        model = PIKKTTypeIIModel(ncol=54, couplings=[0.0059, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            hmc.run(
                model,
                step_size=0.01,
                nsteps=500,
                niters=300,
                output=tmp,
                name="FindCrit",
                dry_run=True,
            )
            expected = (
                Path(tmp)
                / "FindCrit_pikkt4d_type2_g0.0059_omega1.0_N54_dt0.01_steps500_iters300"
            )
            self.assertTrue(expected.is_dir())

    def test_hmc_run_uses_incremented_directory_when_run_exists(self):
        model = PIKKTTypeIIModel(ncol=50, couplings=[0.005, 1.0], bosonic=True)
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(3):
                hmc.run(
                    model,
                    step_size=1.0,
                    nsteps=700,
                    niters=300,
                    output=tmp,
                    name="Run",
                    dry_run=True,
                )

            base = (
                Path(tmp)
                / "Run_bospikkt4d_type2_g0.005_omega1.0_N50_dt1.0_steps700_iters300"
            )
            self.assertTrue(base.is_dir())
            self.assertTrue(Path(f"{base}_run2").is_dir())
            self.assertTrue(Path(f"{base}_run3").is_dir())

    def test_hmc_forced_fresh_run_still_uses_incremented_directory(self):
        model = PIKKTTypeIIModel(ncol=2, couplings=[0.0058, 1.0], bosonic=True, spin=0)
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                hmc.run(
                    model,
                    step_size=0.05,
                    nsteps=1000,
                    niters=300,
                    output=tmp,
                    name="Run",
                    force=True,
                    dry_run=True,
                )

            base = (
                Path(tmp)
                / "Run_bospikkt4d_type2_g0.0058_omega1.0_N2_dt0.05_steps1000_iters300"
            )
            self.assertTrue(base.is_dir())
            self.assertTrue(Path(f"{base}_run2").is_dir())


class TestYangMills(unittest.TestCase):
    """yangmills — from batch.sh manyCouplings runs."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex64")

    def test_dry_run_d2(self):
        # matrix-hmc-track --model yangmills --ncol 80 --nmat 2 --coupling 100 --step-size 2 --nsteps 200
        model = YangMillsModel(dim=2, ncol=80, couplings=[100.0])
        self.assertIs(_run(model, step_size=2.0, nsteps=200, niters=600), model)

    def test_dry_run_d3(self):
        # matrix-hmc-track --model yangmills --ncol 80 --nmat 3 --coupling 100 --step-size 2 --nsteps 200
        model = YangMillsModel(dim=3, ncol=80, couplings=[100.0])
        self.assertIs(_run(model, step_size=2.0, nsteps=200, niters=600), model)

    def test_dry_run_d4(self):
        # matrix-hmc-track --model yangmills --ncol 50 --nmat 4 --coupling 150 --step-size 2 --nsteps 100
        model = YangMillsModel(dim=4, ncol=50, couplings=[150.0])
        self.assertIs(_run(model, step_size=2.0, nsteps=100, niters=600), model)


class TestAdjointDet(unittest.TestCase):
    """adjoint_det — from batch.sh commutingNoGamma2 runs."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex64")

    def test_dry_run_nmat4(self):
        # matrix-hmc-track --model adjoint_det --nmat 4 --ncol 40 --coupling 10 --step-size 1 --nsteps 300
        model = AdjointDetModel(dim=4, ncol=40, couplings=[10.0])
        self.assertIs(_run(model, step_size=1.0, nsteps=300, niters=800), model)

    def test_dry_run_nmat6(self):
        # matrix-hmc-track --model adjoint_det --nmat 6 --ncol 40 --coupling 50 --step-size 0.9 --nsteps 350
        model = AdjointDetModel(dim=6, ncol=40, couplings=[50.0])
        self.assertIs(_run(model, step_size=0.9, nsteps=350, niters=800), model)

    def test_dry_run_large_coupling(self):
        # matrix-hmc-track --model adjoint_det --nmat 6 --ncol 40 --coupling 150 --step-size 0.3 --nsteps 500
        model = AdjointDetModel(dim=6, ncol=40, couplings=[150.0])
        self.assertIs(_run(model, step_size=0.3, nsteps=500, niters=800), model)


class TestSUSYYM3D(unittest.TestCase):
    """susyym_3d — not in batch.sh; using typical parameter choices."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex64")

    def test_dry_run_default(self):
        model = SUSYYM3DModel(ncol=10, couplings=[1.0])
        self.assertIs(_run(model, step_size=0.1, nsteps=50), model)

    def test_dry_run_with_mass(self):
        model = SUSYYM3DModel(ncol=12, couplings=[60.0], fermion_mass=1.0, boson_mass=1.0)
        self.assertIs(_run(model, step_size=0.1, nsteps=50, niters=300), model)


class TestOneMatrixModel(unittest.TestCase):
    """1mm — not in batch.sh; using CLI defaults."""

    def setUp(self):
        hmc.configure(device="cpu", precision="complex64")

    def test_dry_run_default(self):
        model = OneMatrixPolynomialModel(ncol=50, couplings=[1.0, -0.5])
        self.assertIs(_run(model, step_size=0.5, nsteps=50), model)


if __name__ == "__main__":
    unittest.main()
