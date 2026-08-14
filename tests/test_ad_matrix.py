import sys
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from matrix_hmc_track.algebra import (
    ad_matrix,
    ad_matrix_real_antisymmetric,
    dagger,
    get_traceless_maps_cached,
    random_hermitian,
)
from matrix_hmc_track.models.pikkt10d import PIKKT10DModel
from matrix_hmc_track.pfaffian import make_skew_symmetric, slogpfaff


def vec_col(mat: torch.Tensor) -> torch.Tensor:
    return mat.transpose(-1, -2).contiguous().reshape(-1)


def unvec_col(vec: torch.Tensor, n: int) -> torch.Tensor:
    return vec.reshape(n, n).transpose(-1, -2).contiguous()


def hermitian_basis_synthesis(n: int, *, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    basis = []
    eye = torch.eye(n, dtype=dtype)
    basis.append(eye / (n ** 0.5))

    for a in range(n - 1):
        gen = torch.zeros((n, n), dtype=dtype)
        scale = ((a + 1) * (a + 2)) ** 0.5
        idx = torch.arange(a + 1)
        gen[idx, idx] = 1.0 / scale
        gen[a + 1, a + 1] = -(a + 1) / scale
        basis.append(gen)

    for i in range(n):
        for j in range(i + 1, n):
            gen = torch.zeros((n, n), dtype=dtype)
            gen[i, j] = 2.0 ** -0.5
            gen[j, i] = 2.0 ** -0.5
            basis.append(gen)

    for i in range(n):
        for j in range(i + 1, n):
            gen = torch.zeros((n, n), dtype=dtype)
            gen[i, j] = 1j * (2.0 ** -0.5)
            gen[j, i] = -1j * (2.0 ** -0.5)
            basis.append(gen)

    return torch.stack([vec_col(gen) for gen in basis], dim=1)


class AdMatrixTests(unittest.TestCase):
    def test_random_hermitian_supports_batches(self) -> None:
        for n in (2, 3, 4):
            single = random_hermitian(n)
            batch = random_hermitian(n, batchsize=5)

            self.assertEqual(single.shape, (n, n))
            self.assertEqual(batch.shape, (5, n, n))

            torch.testing.assert_close(single, dagger(single), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(batch, dagger(batch), rtol=1e-12, atol=1e-12)

            torch.testing.assert_close(
                torch.diagonal(single).sum().real,
                torch.tensor(0.0, dtype=single.real.dtype),
                rtol=0.0,
                atol=1e-12,
            )
            torch.testing.assert_close(
                torch.diagonal(batch, dim1=-2, dim2=-1).sum(-1).real,
                torch.zeros(5, dtype=batch.real.dtype),
                rtol=0.0,
                atol=1e-12,
            )

    def test_pikkt10d_fermion_pfaffian_accepts_traceless_hermitian_state(self) -> None:
        for massless in (False, True):
            model = PIKKT10DModel(ncol=2, couplings=[1.0], massless=massless)
            X = random_hermitian(model.ncol, batchsize=model.nmat)

            sign, log_abs = model.fermion_pfaffian(X)

            self.assertEqual(sign.shape, torch.Size([]))
            self.assertEqual(log_abs.shape, torch.Size([]))
            self.assertEqual(sign.dtype, X.dtype)
            self.assertEqual(log_abs.dtype, X.real.dtype)
            self.assertTrue(torch.isfinite(log_abs))

    def test_slogpfaff_sign_is_stable_for_large_inputs(self) -> None:
        A = make_skew_symmetric(torch.randn(10, 10, dtype=torch.complex128))
        large_A = 1e150 * A

        sign_small, log_abs_small = slogpfaff(A)
        sign_large, log_abs_large = slogpfaff(large_A)

        self.assertTrue(torch.isfinite(sign_small.real if sign_small.is_complex() else sign_small))
        self.assertTrue(torch.isfinite(sign_large.real if sign_large.is_complex() else sign_large))
        self.assertTrue(torch.isfinite(log_abs_small))
        self.assertTrue(torch.isfinite(log_abs_large))
        torch.testing.assert_close(sign_large, sign_small, rtol=1e-12, atol=1e-12)

    def test_matches_explicit_commutator(self) -> None:
        # torch.manual_seed(0)

        for n in (2, 3, 4):
            X = torch.randn(n, n, dtype=torch.complex128)
            A = torch.randn(n, n, dtype=torch.complex128)

            expected = vec_col(X @ A - A @ X)
            actual = ad_matrix(X) @ vec_col(A)

            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_matches_explicit_commutator_batched(self) -> None:
        # torch.manual_seed(1)

        for n in (2, 3, 4):
            batch = 3
            X = torch.randn(batch, n, n, dtype=torch.complex128)
            A = torch.randn(batch, n, n, dtype=torch.complex128)

            expected = torch.stack([vec_col(X[i] @ A[i] - A[i] @ X[i]) for i in range(batch)])
            actual = torch.einsum("bij,bj->bi", ad_matrix(X), torch.stack([vec_col(a) for a in A]))

            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_is_hermitian_for_hermitian_inputs_and_annihilates_identity(self) -> None:
        # torch.manual_seed(2)

        for n in (2, 3, 4):
            X = torch.randn(n, n, dtype=torch.complex128)
            X = (X + dagger(X)) / 2
            adX = ad_matrix(X)

            torch.testing.assert_close(adX, dagger(adX), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(
                adX @ vec_col(torch.eye(n, dtype=torch.complex128)),
                torch.zeros(n * n, dtype=torch.complex128),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_matches_traceless_coordinate_convention(self) -> None:
        # torch.manual_seed(3)

        for n in (2, 3, 4):
            X = torch.randn(n, n, dtype=torch.complex128)
            Q, _ = get_traceless_maps_cached(n, X.device, X.dtype)
            coeffs = torch.randn(n * n - 1, dtype=torch.complex128)

            vecA = Q @ coeffs
            A = unvec_col(vecA, n)

            expected = vec_col(X @ A - A @ X)
            actual = ad_matrix(X) @ vecA

            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_real_antisymmetric_basis_matches_commutator(self) -> None:
        # torch.manual_seed(4)

        for n in (2, 3, 4):
            X = torch.randn(n, n, dtype=torch.complex128)
            X = (X + dagger(X)) / 2
            basis = hermitian_basis_synthesis(n)
            coeffs = torch.randn(n * n, dtype=torch.float64)

            A = unvec_col(basis @ coeffs.to(torch.complex128), n)
            expected = basis.conj().transpose(0, 1) @ vec_col(1j * (X @ A - A @ X))
            actual = ad_matrix_real_antisymmetric(X) @ coeffs

            torch.testing.assert_close(actual, expected.real, rtol=1e-12, atol=1e-12)

    def test_real_antisymmetric_basis_is_skew_and_trace_mode_is_explicit(self) -> None:
        # torch.manual_seed(5)

        for n in (2, 3, 4):
            X = torch.randn(n, n, dtype=torch.complex128)
            X = (X + dagger(X)) / 2

            full = ad_matrix_real_antisymmetric(X)
            traceless = ad_matrix_real_antisymmetric(X, traceless=True)

            self.assertFalse(full.is_complex())
            torch.testing.assert_close(full, -full.transpose(-1, -2), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(full[0], torch.zeros(n * n, dtype=full.dtype), rtol=0.0, atol=1e-12)
            torch.testing.assert_close(full[:, 0], torch.zeros(n * n, dtype=full.dtype), rtol=0.0, atol=1e-12)
            torch.testing.assert_close(traceless, full[1:, 1:], rtol=1e-12, atol=1e-12)
    
    def test_real_antisymmetric_det_matches_admatrix_det(self) -> None:
        # torch.manual_seed(6)

        for n in (2, 3, 4):
            X = torch.randn(n, n, dtype=torch.complex128)
            X = (X + dagger(X)) / 2

            adX = 1j * ad_matrix(X)
            adX_real = ad_matrix_real_antisymmetric(X)

            expected_det = torch.linalg.det(adX)
            actual_det = torch.linalg.det(adX_real)

            expected_moments = [(torch.trace(torch.matrix_power(adX, k)).real) for k in range(1,6)]
            actual_moments = [(torch.trace(torch.matrix_power(adX_real, k))) for k in range(1,6)]

            torch.testing.assert_close(actual_det.real, expected_det.real, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(actual_moments, expected_moments, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()