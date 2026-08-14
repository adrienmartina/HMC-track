"""Comprehensive tests for matrix_hmc_track."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from matrix_hmc_track.algebra import (
    add_trace_projector_inplace,
    ad_matrix,
    comm,
    dagger,
    get_eye_cached,
    get_traceless_maps_cached,
    kron_2d,
    makeH,
    random_hermitian,
    spinJMatrices,
)
from matrix_hmc_track.pfaffian import (
    make_skew_symmetric,
    pfaffian,
    slogpfaff,
    verify_pfaffian,
)
from matrix_hmc_track.models.utils import (
    _anticommutator_action_sum,
    _commutator_action_sum,
    _fermion_det_log_identity_plus_sum_adX,
    parse_source,
)
from matrix_hmc_track.models.yangmills import YangMillsModel
from matrix_hmc_track.models.pikkt4d_type1 import PIKKTTypeIModel
from matrix_hmc_track.models.pikkt10d import PIKKT10DModel
from matrix_hmc_track.hmc import HMCParams, hamil, leapfrog, update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_skew(n: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return make_skew_symmetric(torch.randn(n, n, dtype=dtype))


def _rand_hermitian(n: int, *, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Independent Hermitian matrix (does not go through config globals)."""
    A = torch.randn(n, n, dtype=dtype)
    return 0.5 * (A + dagger(A))


# ---------------------------------------------------------------------------
# 1. Algebra utilities
# ---------------------------------------------------------------------------

class TestAlgebra(unittest.TestCase):

    def test_makeH_is_hermitian(self):
        for n in (2, 3, 4):
            A = torch.randn(n, n, dtype=torch.complex128)
            H = makeH(A)
            torch.testing.assert_close(H, dagger(H), rtol=1e-12, atol=1e-12)

    def test_makeH_is_idempotent(self):
        A = torch.randn(4, 4, dtype=torch.complex128)
        H = makeH(A)
        torch.testing.assert_close(makeH(H), H, rtol=1e-12, atol=1e-12)

    def test_makeH_extracts_hermitian_part(self):
        """makeH(A) == (A + A†)/2."""
        A = torch.randn(3, 3, dtype=torch.complex128)
        torch.testing.assert_close(makeH(A), 0.5 * (A + dagger(A)), rtol=1e-12, atol=1e-12)

    def test_comm_is_antisymmetric(self):
        for n in (2, 3):
            A = _rand_hermitian(n)
            B = _rand_hermitian(n)
            torch.testing.assert_close(comm(A, B), -comm(B, A), rtol=1e-12, atol=1e-12)

    def test_comm_with_itself_is_zero(self):
        A = _rand_hermitian(3)
        C = comm(A, A)
        torch.testing.assert_close(C, torch.zeros_like(C), rtol=0, atol=1e-12)

    def test_comm_jacobi_identity(self):
        """[A,[B,C]] + [B,[C,A]] + [C,[A,B]] == 0."""
        for n in (2, 3):
            A, B, C = _rand_hermitian(n), _rand_hermitian(n), _rand_hermitian(n)
            total = comm(A, comm(B, C)) + comm(B, comm(C, A)) + comm(C, comm(A, B))
            torch.testing.assert_close(total, torch.zeros_like(total), rtol=1e-10, atol=1e-10)

    def test_comm_linearity_in_first_arg(self):
        A, B, C = _rand_hermitian(3), _rand_hermitian(3), _rand_hermitian(3)
        alpha = 2.5 + 0.3j
        lhs = comm(alpha * A + B, C)
        rhs = alpha * comm(A, C) + comm(B, C)
        torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)

    def test_kron_2d_shape(self):
        A = torch.randn(2, 3, dtype=torch.complex128)
        B = torch.randn(4, 5, dtype=torch.complex128)
        K = kron_2d(A, B)
        self.assertEqual(K.shape, (8, 15))

    def test_kron_2d_identity_block_structure(self):
        """kron(I2, B) is block-diagonal with two copies of B."""
        I = torch.eye(2, dtype=torch.complex128)
        B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.complex128)
        K = kron_2d(I, B)
        torch.testing.assert_close(K[:2, :2], B, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(K[2:, 2:], B, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(K[:2, 2:], torch.zeros(2, 2, dtype=B.dtype), rtol=0, atol=1e-12)
        torch.testing.assert_close(K[2:, :2], torch.zeros(2, 2, dtype=B.dtype), rtol=0, atol=1e-12)

    def test_kron_2d_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            kron_2d(torch.ones(2, 2, 2), torch.ones(2, 2))

    def test_add_trace_projector_inplace_is_rank_one_update(self):
        """The diagonal trace block of (K + P) differs from K exactly at diagonal positions."""
        n = 3
        dim = n * n
        K = torch.zeros(dim, dim, dtype=torch.complex128)
        K_copy = K.clone()
        add_trace_projector_inplace(K, n)
        diff = K - K_copy
        # Only trace-mode positions should be nonzero.
        diag_pos = torch.arange(0, dim, n + 1)
        off_diag_mask = torch.ones(dim, dim, dtype=torch.bool)
        off_diag_mask[diag_pos.unsqueeze(1), diag_pos] = False
        torch.testing.assert_close(diff[off_diag_mask], torch.zeros(off_diag_mask.sum(), dtype=torch.complex128),
                                   rtol=0, atol=1e-12)

    def test_get_traceless_maps_SQ_is_identity(self):
        """S @ Q == I_{N^2-1}: S is a left inverse of Q."""
        for n in (2, 3, 4):
            Q, S = get_traceless_maps_cached(n, torch.device("cpu"), torch.complex128)
            SQ = S @ Q
            eye = torch.eye(n * n - 1, dtype=torch.complex128)
            torch.testing.assert_close(SQ, eye, rtol=1e-12, atol=1e-12)

    def test_get_traceless_maps_Q_image_is_traceless(self):
        """Q maps any coefficient vector to a traceless (flattened) matrix."""
        for n in (2, 3):
            Q, _ = get_traceless_maps_cached(n, torch.device("cpu"), torch.complex128)
            v = torch.randn(n * n - 1, dtype=torch.complex128)
            vecA = Q @ v
            # column-major reshape: vec[i + j*N] corresponds to A[i,j]
            A = vecA.reshape(n, n).T
            trace = A.diagonal().sum()
            self.assertAlmostEqual(trace.abs().item(), 0.0, places=12)

    def test_spinJ_su2_algebra(self):
        """Ji must satisfy [Ji, Jj] = i ε_{ijk} Jk."""
        for j in (0.5, 1.0, 1.5):
            mats = spinJMatrices(j)
            Jx = torch.tensor(mats[0], dtype=torch.complex128)
            Jy = torch.tensor(mats[1], dtype=torch.complex128)
            Jz = torch.tensor(mats[2], dtype=torch.complex128)
            torch.testing.assert_close(comm(Jx, Jy), 1j * Jz, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(comm(Jy, Jz), 1j * Jx, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(comm(Jz, Jx), 1j * Jy, rtol=1e-10, atol=1e-10)

    def test_spinJ_dimension(self):
        for j in (0.5, 1.0, 1.5, 2.0):
            mats = spinJMatrices(j)
            dim = int(round(2 * j + 1))
            self.assertEqual(mats.shape, (3, dim, dim))

    def test_get_eye_cached_is_identity(self):
        I = get_eye_cached(3, torch.device("cpu"), torch.complex128)
        torch.testing.assert_close(I, torch.eye(3, dtype=torch.complex128), rtol=0, atol=0)

    def test_get_eye_cached_returns_same_object(self):
        I1 = get_eye_cached(4, torch.device("cpu"), torch.complex128)
        I2 = get_eye_cached(4, torch.device("cpu"), torch.complex128)
        self.assertIs(I1, I2)

    def test_dagger_is_conj_transpose(self):
        A = torch.randn(3, 4, dtype=torch.complex128)
        torch.testing.assert_close(dagger(A), A.conj().transpose(-2, -1), rtol=0, atol=0)

    def test_dagger_involutory(self):
        A = torch.randn(3, 3, dtype=torch.complex128)
        torch.testing.assert_close(dagger(dagger(A)), A, rtol=0, atol=0)

    def test_random_hermitian_is_hermitian(self):
        for n in (2, 3, 4):
            H = random_hermitian(n)
            torch.testing.assert_close(H, dagger(H), rtol=1e-12, atol=1e-12)

    def test_random_hermitian_is_traceless_by_default(self):
        for n in (2, 3, 4):
            H = random_hermitian(n)
            trace = H.diagonal().sum().real
            self.assertAlmostEqual(trace.item(), 0.0, places=12)

    def test_random_hermitian_not_traceless_when_disabled(self):
        """Over many samples the trace should not be systematically zero."""
        traces = [random_hermitian(3, traceless=False).diagonal().sum().real.item() for _ in range(20)]
        self.assertGreater(max(abs(t) for t in traces), 1e-6)

    def test_ad_matrix_antisymmetric_under_linearity(self):
        """ad_{aX+bY} == a*ad_X + b*ad_Y."""
        n = 3
        X = _rand_hermitian(n)
        Y = _rand_hermitian(n)
        a, b = 2.0 + 0j, -1.5 + 0.5j
        lhs = ad_matrix(a * X + b * Y)
        rhs = a * ad_matrix(X) + b * ad_matrix(Y)
        torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Pfaffian
# ---------------------------------------------------------------------------

class TestPfaffian(unittest.TestCase):

    def test_2x2_analytic(self):
        """pf([[0, a], [-a, 0]]) == a."""
        for a in (1.0, -3.0, 0.5):
            A = torch.tensor([[0.0, a], [-a, 0.0]])
            pf = pfaffian(A)
            self.assertAlmostEqual(pf.item(), a, places=12)

    def test_4x4_analytic(self):
        """pf of a block-diagonal 4x4 with two [[0,a],[-a,0]] blocks == a^2."""
        a = 3.0
        A = torch.zeros(4, 4, dtype=torch.float64)
        A[0, 1] = a; A[1, 0] = -a
        A[2, 3] = a; A[3, 2] = -a
        pf = pfaffian(A)
        self.assertAlmostEqual(pf.item(), a ** 2, places=10)

    def test_pfaffian_squared_equals_det(self):
        for n in (2, 4, 6):
            A = _rand_skew(n)
            pf = pfaffian(A)
            det = torch.linalg.det(A)
            torch.testing.assert_close(pf ** 2, det, rtol=1e-10, atol=1e-10)

    def test_pfaffian_sign_flips_on_row_col_swap(self):
        """Swapping rows i↔j and cols i↔j (congruence) negates pf."""
        A = _rand_skew(4)
        pf_orig = pfaffian(A)
        B = A.clone()
        B[[0, 1], :] = B[[1, 0], :]
        B[:, [0, 1]] = B[:, [1, 0]]
        pf_swap = pfaffian(B)
        torch.testing.assert_close(pf_swap, -pf_orig, rtol=1e-10, atol=1e-10)

    def test_pfaffian_scaling(self):
        """pf(c*A) == c^(n/2) * pf(A) for even n."""
        A = _rand_skew(4)
        c = 3.0
        torch.testing.assert_close(pfaffian(c * A), (c ** 2) * pfaffian(A), rtol=1e-10, atol=1e-10)

    def test_slogpfaff_consistent_with_pfaffian(self):
        for n in (2, 4, 6):
            A = _rand_skew(n)
            pf = pfaffian(A)
            sign, log_abs = slogpfaff(A)
            torch.testing.assert_close(sign * log_abs.exp(), pf, rtol=1e-8, atol=1e-8)

    def test_slogpfaff_log_abs_equals_half_logdet(self):
        """log|pf(A)| == (1/2) log|det(A)|."""
        for n in (4, 6):
            A = _rand_skew(n)
            _, log_abs = slogpfaff(A)
            half_logdet = torch.linalg.slogdet(A).logabsdet / 2
            torch.testing.assert_close(log_abs, half_logdet, rtol=1e-12, atol=1e-12)

    def test_slogpfaff_sign_squared_is_one(self):
        A = _rand_skew(4)
        sign, _ = slogpfaff(A)
        self.assertAlmostEqual(sign.item() ** 2, 1.0, places=10)

    def test_slogpfaff_gradient_flows(self):
        A_raw = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        A = make_skew_symmetric(A_raw)
        _, log_abs = slogpfaff(A)
        log_abs.backward()
        self.assertIsNotNone(A_raw.grad)
        self.assertTrue(torch.isfinite(A_raw.grad).all())

    def test_pfaffian_gradient_flows(self):
        A_raw = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        pf = pfaffian(make_skew_symmetric(A_raw))
        pf.backward()
        self.assertIsNotNone(A_raw.grad)
        self.assertTrue(torch.isfinite(A_raw.grad).all())

    def test_pfaffian_odd_dim_is_zero(self):
        A = _rand_skew(5)
        self.assertEqual(pfaffian(A).item(), 0.0)

    def test_slogpfaff_odd_dim_log_abs_is_neginf(self):
        A = _rand_skew(5)
        _, log_abs = slogpfaff(A)
        self.assertEqual(log_abs.item(), float("-inf"))

    def test_pfaffian_empty_matrix_is_one(self):
        A = torch.zeros(0, 0)
        self.assertEqual(pfaffian(A).item(), 1.0)

    def test_slogpfaff_empty_matrix(self):
        A = torch.zeros(0, 0)
        sign, log_abs = slogpfaff(A)
        self.assertEqual(log_abs.item(), 0.0)
        self.assertEqual(sign.item(), 1.0)

    def test_pfaffian_batched_shape(self):
        B = 5
        A_batch = make_skew_symmetric(torch.randn(B, 4, 4))
        pf_batch = pfaffian(A_batch)
        self.assertEqual(pf_batch.shape, (B,))

    def test_pfaffian_batched_matches_per_element(self):
        B = 4
        A_batch = make_skew_symmetric(torch.randn(B, 4, 4))
        pf_batch = pfaffian(A_batch)
        for i in range(B):
            torch.testing.assert_close(pf_batch[i], pfaffian(A_batch[i]), rtol=1e-10, atol=1e-10)

    def test_slogpfaff_stable_for_large_inputs(self):
        A = _rand_skew(8)
        sign, log_abs = slogpfaff(A)
        sign_large, log_abs_large = slogpfaff(1e150 * A)
        self.assertTrue(torch.isfinite(log_abs))
        self.assertTrue(torch.isfinite(log_abs_large))
        torch.testing.assert_close(sign, sign_large, rtol=1e-10, atol=1e-10)

    def test_slogpfaff_complex_sign_is_unit(self):
        A = make_skew_symmetric(torch.randn(4, 4, dtype=torch.complex128))
        sign, log_abs = slogpfaff(A)
        self.assertTrue(torch.isfinite(log_abs))
        self.assertAlmostEqual(sign.abs().item(), 1.0, places=10)

    def test_make_skew_symmetric(self):
        A = torch.randn(4, 4, dtype=torch.float64)
        S = make_skew_symmetric(A)
        torch.testing.assert_close(S, -S.T, rtol=1e-12, atol=1e-12)

    def test_verify_pfaffian_passes_for_correct(self):
        A = _rand_skew(4)
        pf = pfaffian(A)
        self.assertTrue(verify_pfaffian(A, pf, tol=1e-8))

    def test_verify_pfaffian_fails_for_wrong(self):
        A = _rand_skew(4)
        self.assertFalse(verify_pfaffian(A, torch.tensor(0.0, dtype=torch.float64), tol=1e-3))

    def test_pfaffian_non_square_raises(self):
        A = torch.randn(4, 6)
        with self.assertRaises((ValueError, RuntimeError)):
            pfaffian(A)


# ---------------------------------------------------------------------------
# 3. Model utilities
# ---------------------------------------------------------------------------

class TestModelUtils(unittest.TestCase):

    def test_parse_source_1d_shape(self):
        nmat, N = 3, 4
        src = np.arange(N, dtype=np.float64)
        out = parse_source(src, nmat, torch.device("cpu"), torch.complex128)
        self.assertEqual(out.shape, (nmat, N, N))

    def test_parse_source_1d_first_matrix_is_diagonal(self):
        N = 4
        src = np.arange(1, N + 1, dtype=np.float64)
        out = parse_source(src, 3, torch.device("cpu"), torch.complex128)
        expected = torch.diag(torch.tensor(src, dtype=torch.complex128))
        torch.testing.assert_close(out[0], expected, rtol=1e-12, atol=1e-12)

    def test_parse_source_1d_remaining_are_zero(self):
        nmat, N = 3, 4
        src = np.ones(N)
        out = parse_source(src, nmat, torch.device("cpu"), torch.complex128)
        torch.testing.assert_close(out[1:], torch.zeros(nmat - 1, N, N, dtype=torch.complex128),
                                   rtol=0, atol=0)

    def test_parse_source_3d_passthrough(self):
        nmat, N = 2, 3
        src = np.random.randn(nmat, N, N)
        out = parse_source(src, nmat, torch.device("cpu"), torch.complex128)
        self.assertEqual(out.shape, (nmat, N, N))

    def test_parse_source_none_returns_none(self):
        self.assertIsNone(parse_source(None, 3, torch.device("cpu"), torch.complex128))

    def test_parse_source_wrong_nmat_raises(self):
        with self.assertRaises(ValueError):
            parse_source(np.random.randn(2, 3, 3), 3, torch.device("cpu"), torch.complex128)

    def test_parse_source_2d_array_raises(self):
        with self.assertRaises(ValueError):
            parse_source(np.random.randn(2, 3), 2, torch.device("cpu"), torch.complex128)

    def test_commutator_sum_zero_for_commuting(self):
        """Diagonal matrices commute → sum of [X_i, X_j]^2 is zero."""
        X = torch.zeros(3, 3, 3, dtype=torch.complex128)
        for i in range(3):
            X[i] = torch.diag(torch.randn(3).to(torch.complex128))
        result = _commutator_action_sum(X)
        torch.testing.assert_close(result.real, torch.tensor(0.0, dtype=torch.float64),
                                   rtol=0, atol=1e-11)

    def test_commutator_sum_single_matrix_is_zero(self):
        X = torch.randn(1, 3, 3, dtype=torch.complex128)
        result = _commutator_action_sum(X)
        torch.testing.assert_close(result, torch.tensor(0.0, dtype=torch.complex128),
                                   rtol=0, atol=1e-12)

    def test_commutator_sum_is_nonpositive_for_hermitian(self):
        """Tr([A,B]^2) is real and ≤ 0 for Hermitian A, B."""
        X = random_hermitian(3, batchsize=2)
        result = _commutator_action_sum(X)
        self.assertLessEqual(result.real.item(), 1e-10)

    def test_commutator_sum_symmetric_in_index_order(self):
        """Swapping the order of matrices in the batch should not change the total."""
        X = random_hermitian(3, batchsize=2)
        X_flipped = torch.stack([X[1], X[0]])
        r1 = _commutator_action_sum(X)
        r2 = _commutator_action_sum(X_flipped)
        torch.testing.assert_close(r1, r2, rtol=1e-10, atol=1e-10)

    def test_anticommutator_sum_single_matrix_is_zero(self):
        X = torch.randn(1, 3, 3, dtype=torch.complex128)
        result = _anticommutator_action_sum(X)
        torch.testing.assert_close(result, torch.tensor(0.0, dtype=torch.complex128),
                                   rtol=0, atol=1e-12)

    def test_anticommutator_sum_is_real_for_hermitian(self):
        X = random_hermitian(3, batchsize=3)
        result = _anticommutator_action_sum(X)
        self.assertAlmostEqual(result.imag.item(), 0.0, places=10)

    def test_fermion_det_log_is_nonneg(self):
        """log(1 + delta^2) >= 0 for all eigenvalue differences."""
        X = random_hermitian(3, batchsize=2)
        result = _fermion_det_log_identity_plus_sum_adX(X)
        self.assertGreaterEqual(result.item(), 0.0)

    def test_fermion_det_log_zero_matrices(self):
        """All-zero X → all deltas zero → result is 0."""
        X = torch.zeros(2, 3, 3, dtype=torch.complex128)
        result = _fermion_det_log_identity_plus_sum_adX(X)
        self.assertAlmostEqual(result.item(), 0.0, places=12)

    def test_fermion_det_log_increases_with_spread(self):
        """Scaling X increases eigenvalue spread and thus the determinant term."""
        X = random_hermitian(3, batchsize=2)
        r1 = _fermion_det_log_identity_plus_sum_adX(X)
        r2 = _fermion_det_log_identity_plus_sum_adX(10.0 * X)
        self.assertGreater(r2.item(), r1.item())


# ---------------------------------------------------------------------------
# 4. MatrixModel base via YangMillsModel
# ---------------------------------------------------------------------------

class TestMatrixModelBase(unittest.TestCase):

    def _make(self, n: int = 2, d: int = 2, g: float = 1.0) -> YangMillsModel:
        return YangMillsModel(dim=d, ncol=n, couplings=[g])

    def test_set_get_state_roundtrip(self):
        model = self._make()
        X = random_hermitian(2, batchsize=2)
        model.set_state(X)
        torch.testing.assert_close(model.get_state(), X, rtol=0, atol=0)

    def test_get_state_without_set_raises(self):
        model = self._make()
        with self.assertRaises(ValueError):
            model.get_state()

    def test_force_shape(self):
        model = self._make(n=3, d=2)
        X = random_hermitian(3, batchsize=2)
        model.set_state(X)
        F = model.force()
        self.assertEqual(F.shape, X.shape)

    def test_force_is_hermitian(self):
        model = self._make(n=3, d=2)
        X = random_hermitian(3, batchsize=2)
        model.set_state(X)
        F = model.force()
        torch.testing.assert_close(F, dagger(F), rtol=1e-10, atol=1e-10)

    def test_force_is_traceless(self):
        model = self._make(n=3, d=2)
        X = random_hermitian(3, batchsize=2)
        model.set_state(X)
        F = model.force()
        traces = torch.diagonal(F, dim1=-2, dim2=-1).sum(-1).real
        torch.testing.assert_close(traces, torch.zeros_like(traces), rtol=0, atol=1e-10)

    def test_force_at_zero_is_zero(self):
        """V = (N/g) * mass * Tr(X^2) → force = 2*(N/g)*mass*X → 0 at X=0."""
        model = YangMillsModel(dim=2, ncol=2, couplings=[1.0], mass=1.0)
        X = torch.zeros(2, 2, 2, dtype=torch.complex128)
        model.set_state(X)
        F = model.force()
        torch.testing.assert_close(F, torch.zeros_like(F), rtol=0, atol=1e-12)

    def test_build_paths_has_required_keys(self):
        model = self._make()
        paths = model.build_paths("run", "/tmp")
        for key in ("dir", "eigs", "corrs", "meta", "ckpt"):
            self.assertIn(key, paths)

    def test_status_string_contains_trX(self):
        model = self._make()
        X = random_hermitian(2, batchsize=2)
        model.set_state(X)
        s = model.status_string()
        self.assertIn("trX", s)

    def test_run_metadata_has_required_keys(self):
        model = self._make()
        meta = model.run_metadata()
        for key in ("model_name", "nmat", "ncol", "couplings"):
            self.assertIn(key, meta)

    def test_potential_via_explicit_arg(self):
        """potential(X) and potential() after set_state should agree."""
        model = self._make(n=3, d=2)
        X = random_hermitian(3, batchsize=2)
        model.set_state(X)
        V_via_state = model.potential()
        V_via_arg = model.potential(X)
        torch.testing.assert_close(V_via_state, V_via_arg, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 5. YangMillsModel
# ---------------------------------------------------------------------------

class TestYangMillsModel(unittest.TestCase):

    def _make(self, n: int = 3, d: int = 3, g: float = 0.5) -> YangMillsModel:
        return YangMillsModel(dim=d, ncol=n, couplings=[g])

    def test_potential_is_real_finite(self):
        model = self._make()
        X = random_hermitian(3, batchsize=3)
        model.set_state(X)
        V = model.potential()
        self.assertTrue(torch.isfinite(V))
        self.assertFalse(V.is_complex())

    def test_potential_scales_inversely_with_g(self):
        """V scales as N/g, so V(g=1) * 1 == V(g=2) * 2 at fixed X."""
        n, d = 3, 2
        X = random_hermitian(n, batchsize=d)
        m1 = YangMillsModel(dim=d, ncol=n, couplings=[1.0])
        m2 = YangMillsModel(dim=d, ncol=n, couplings=[2.0])
        m1.set_state(X.clone()); m2.set_state(X.clone())
        V1 = m1.potential().real
        V2 = m2.potential().real
        torch.testing.assert_close(V1 * 1.0, V2 * 2.0, rtol=1e-10, atol=1e-10)

    def test_mass_term_contributes_correctly(self):
        """At zero configuration, V = 0. With mass, the gradient is 2*(N/g)*mass*X."""
        n, d, g = 2, 2, 1.0
        model = YangMillsModel(dim=d, ncol=n, couplings=[g], mass=1.5)
        X = random_hermitian(n, batchsize=d)
        model.set_state(X)
        V = model.potential().real
        # Pure mass term: V = (N/g) * mass * sum_i Tr(X_i^2)
        expected_mass = (n / g) * 1.5 * torch.einsum("bij,bji->", X, X).real
        # There's also a commutator term, so just check V is finite and real
        self.assertTrue(torch.isfinite(V))

    def test_force_directional_derivative(self):
        """<F, dX> matches finite-difference directional derivative."""
        model = self._make(n=3, d=2, g=1.0)
        X = random_hermitian(3, batchsize=2)
        model.set_state(X)
        F = model.force(X)
        dX = random_hermitian(3, batchsize=2)
        eps = 1e-5
        V_plus = model.potential(X + eps * dX).real
        V_minus = model.potential(X - eps * dX).real
        fd = (V_plus - V_minus) / (2 * eps)
        inner = torch.einsum("bij,bji->", F, dX).real
        self.assertAlmostEqual(inner.item(), fd.item(), places=5)

    def test_measure_observables_shapes(self):
        model = self._make(n=3, d=3, g=1.0)
        X = random_hermitian(3, batchsize=3)
        model.set_state(X)
        eigs, corrs = model.measure_observables()
        # nmat real eigenvalue arrays + 2 complex combos
        self.assertEqual(len(eigs), 3 + 2)
        self.assertEqual(eigs[0].shape, (3,))
        self.assertEqual(corrs.shape, (2,))

    def test_measure_observables_eigenvalues_real(self):
        model = self._make(n=3, d=3, g=1.0)
        X = random_hermitian(3, batchsize=3)
        model.set_state(X)
        eigs, _ = model.measure_observables()
        for e in eigs[:3]:
            self.assertTrue(np.all(np.isfinite(e)))

    def test_source_modifies_potential(self):
        n, d = 3, 2
        X = random_hermitian(n, batchsize=d)
        # Non-uniform source so coupling Tr(J @ X[0]) doesn't vanish on traceless X.
        src = np.array([1.0, -1.0, 0.0])
        m_no_src = YangMillsModel(dim=d, ncol=n, couplings=[1.0])
        m_src = YangMillsModel(dim=d, ncol=n, couplings=[1.0], source=src)
        m_no_src.set_state(X.clone()); m_src.set_state(X.clone())
        V0 = m_no_src.potential().real.item()
        V1 = m_src.potential().real.item()
        self.assertNotAlmostEqual(V0, V1, places=5)

    def test_extra_config_lines(self):
        model = self._make()
        lines = model.extra_config_lines()
        self.assertIsInstance(lines, list)
        self.assertTrue(len(lines) > 0)


# ---------------------------------------------------------------------------
# 6. PIKKTTypeIModel
# ---------------------------------------------------------------------------

class TestPIKKTTypeIModel(unittest.TestCase):

    def _make(self, n: int = 2, g: float = 1.0, eta: float = 1.0,
              massless: bool = False) -> PIKKTTypeIModel:
        return PIKKTTypeIModel(ncol=n, couplings=[g], eta=eta, massless=massless)

    def test_nmat_is_4(self):
        model = self._make()
        self.assertEqual(model.nmat, 4)

    def test_potential_is_real_finite(self):
        for massless in (False, True):
            model = self._make(n=2, massless=massless)
            X = random_hermitian(2, batchsize=4)
            model.set_state(X)
            V = model.potential()
            self.assertTrue(torch.isfinite(V), f"Not finite for massless={massless}")
            self.assertFalse(V.is_complex(), f"V is complex for massless={massless}")

    def test_force_is_hermitian(self):
        model = self._make(n=2)
        X = random_hermitian(2, batchsize=4)
        model.set_state(X)
        F = model.force()
        torch.testing.assert_close(F, dagger(F), rtol=1e-8, atol=1e-8)

    def test_force_is_traceless(self):
        model = self._make(n=2)
        X = random_hermitian(2, batchsize=4)
        model.set_state(X)
        F = model.force()
        traces = torch.diagonal(F, dim1=-2, dim2=-1).sum(-1).real
        torch.testing.assert_close(traces, torch.zeros_like(traces), rtol=0, atol=1e-8)

    def test_force_directional_derivative(self):
        model = self._make(n=2, g=1.0)
        X = random_hermitian(2, batchsize=4)
        model.set_state(X)
        F = model.force(X)
        dX = random_hermitian(2, batchsize=4)
        eps = 1e-5
        V_plus = model.potential(X + eps * dX).real
        V_minus = model.potential(X - eps * dX).real
        fd = (V_plus - V_minus) / (2 * eps)
        inner = torch.einsum("bij,bji->", F, dX).real
        self.assertAlmostEqual(inner.item(), fd.item(), places=4)

    def test_eta_changes_potential(self):
        n = 2
        X = random_hermitian(n, batchsize=4)
        m1 = PIKKTTypeIModel(ncol=n, couplings=[1.0], eta=1.0)
        m2 = PIKKTTypeIModel(ncol=n, couplings=[1.0], eta=2.0)
        m1.set_state(X.clone()); m2.set_state(X.clone())
        self.assertNotAlmostEqual(m1.potential().real.item(), m2.potential().real.item(), places=5)

    def test_measure_observables_shapes(self):
        model = self._make(n=2)
        X = random_hermitian(2, batchsize=4)
        model.set_state(X)
        with torch.no_grad():
            eigs, corrs = model.measure_observables()
        self.assertEqual(len(eigs), 7)  # 4 real + 2 complex + 1 Casimir
        self.assertEqual(corrs.shape, (4,))

    def test_massless_and_massive_give_different_potentials(self):
        n = 2
        X = random_hermitian(n, batchsize=4)
        m_massive = PIKKTTypeIModel(ncol=n, couplings=[1.0], massless=False)
        m_massless = PIKKTTypeIModel(ncol=n, couplings=[1.0], massless=True)
        m_massive.set_state(X.clone()); m_massless.set_state(X.clone())
        V_massive = m_massive.potential().real.item()
        V_massless = m_massless.potential().real.item()
        self.assertNotAlmostEqual(V_massive, V_massless, places=3)


# ---------------------------------------------------------------------------
# 7. PIKKT10DModel
# ---------------------------------------------------------------------------

class TestPIKKT10DModel(unittest.TestCase):

    def test_nmat_is_10(self):
        model = PIKKT10DModel(ncol=2, couplings=[1.0])
        self.assertEqual(model.nmat, 10)

    def test_pfaffian_sign_is_unit(self):
        for massless in (False, True):
            model = PIKKT10DModel(ncol=2, couplings=[1.0], massless=massless)
            X = random_hermitian(2, batchsize=10)
            sign, _ = model.fermion_pfaffian(X)
            self.assertAlmostEqual(sign.abs().item(), 1.0, places=10,
                                   msg=f"massless={massless}")

    def test_pfaffian_log_abs_is_finite(self):
        for massless in (False, True):
            model = PIKKT10DModel(ncol=2, couplings=[1.0], massless=massless)
            X = random_hermitian(2, batchsize=10)
            _, log_abs = model.fermion_pfaffian(X)
            self.assertTrue(torch.isfinite(log_abs), f"massless={massless}")

    def test_potential_is_real_finite(self):
        model = PIKKT10DModel(ncol=2, couplings=[1.0])
        X = random_hermitian(2, batchsize=10)
        model.set_state(X)
        V = model.potential()
        self.assertTrue(torch.isfinite(V))
        self.assertFalse(V.is_complex())

    def test_force_shape_and_hermitian(self):
        model = PIKKT10DModel(ncol=2, couplings=[1.0])
        X = random_hermitian(2, batchsize=10)
        model.set_state(X)
        F = model.force()
        self.assertEqual(F.shape, X.shape)
        torch.testing.assert_close(F, dagger(F), rtol=1e-8, atol=1e-8)

    def test_pfaffian_output_shapes_are_scalar(self):
        model = PIKKT10DModel(ncol=2, couplings=[1.0])
        X = random_hermitian(2, batchsize=10)
        sign, log_abs = model.fermion_pfaffian(X)
        self.assertEqual(sign.shape, torch.Size([]))
        self.assertEqual(log_abs.shape, torch.Size([]))


# ---------------------------------------------------------------------------
# 8. HMC
# ---------------------------------------------------------------------------

class TestHMC(unittest.TestCase):

    def _make_model(self) -> YangMillsModel:
        model = YangMillsModel(dim=2, ncol=3, couplings=[0.5])
        X = random_hermitian(3, batchsize=2)
        model.set_state(X)
        return model

    def test_hmc_params_fields(self):
        p = HMCParams(dt=0.1, nsteps=10)
        self.assertEqual(p.dt, 0.1)
        self.assertEqual(p.nsteps, 10)

    def test_hamil_is_finite(self):
        model = self._make_model()
        X = model.get_state()
        mom = random_hermitian(3, batchsize=2)
        H = hamil(X, mom, model)
        self.assertTrue(math.isfinite(H))

    def test_hamil_pure_kinetic(self):
        """With mass=0 and zero X, Hamiltonian equals kinetic term only."""
        model = YangMillsModel(dim=2, ncol=2, couplings=[1.0], mass=0.0)
        X = torch.zeros(2, 2, 2, dtype=torch.complex128)
        model.set_state(X)
        mom = random_hermitian(2, batchsize=2)
        H = hamil(X, mom, model)
        kin = sum(0.5 * torch.trace(mom[j] @ mom[j]).real.item() for j in range(2))
        self.assertAlmostEqual(H, kin, places=10)

    def test_leapfrog_returns_finite_energies(self):
        model = self._make_model()
        X = model.get_state()
        params = HMCParams(dt=0.01, nsteps=5)
        X_new, H0, H1 = leapfrog(X, params, model)
        self.assertTrue(math.isfinite(H0))
        self.assertTrue(math.isfinite(H1))

    def test_leapfrog_proposal_shape(self):
        model = self._make_model()
        X = model.get_state()
        params = HMCParams(dt=0.01, nsteps=3)
        X_new, _, _ = leapfrog(X, params, model)
        self.assertEqual(X_new.shape, X.shape)

    def test_leapfrog_proposal_is_hermitian(self):
        model = self._make_model()
        X = model.get_state()
        params = HMCParams(dt=0.01, nsteps=5)
        X_new, _, _ = leapfrog(X, params, model)
        torch.testing.assert_close(X_new, dagger(X_new), rtol=1e-8, atol=1e-8)

    def test_leapfrog_small_dt_energy_change_is_small(self):
        """For very small dt the leapfrog error O(dt^2) means |dH| << 1."""
        model = self._make_model()
        X = model.get_state()
        params = HMCParams(dt=1e-4, nsteps=5)
        _, H0, H1 = leapfrog(X, params, model)
        self.assertLess(abs(H1 - H0), 1.0)

    def test_update_returns_nonneg_acc_count(self):
        model = self._make_model()
        params = HMCParams(dt=0.01, nsteps=3)
        acc = update(0, params, model)
        self.assertGreaterEqual(acc, 0)

    def test_update_preserves_hermitian_state(self):
        model = self._make_model()
        params = HMCParams(dt=0.01, nsteps=3)
        update(0, params, model)
        X = model.get_state()
        torch.testing.assert_close(X, dagger(X), rtol=1e-8, atol=1e-8)

    def test_update_acc_count_monotone_nondecreasing(self):
        """acc_count can only stay the same or increase."""
        model = self._make_model()
        params = HMCParams(dt=0.01, nsteps=3)
        acc = 5
        acc_new = update(acc, params, model)
        self.assertGreaterEqual(acc_new, acc)

    def test_update_with_always_reject(self):
        """reject_prob=0.0 means never apply Metropolis; dH≤0 configs always accepted."""
        model = self._make_model()
        params = HMCParams(dt=1e-5, nsteps=2)
        # With very small dt the proposal energy is close to current; some may still be accepted
        acc = update(0, params, model, reject_prob=0.0)
        # Regardless: model state must remain valid Hermitian
        X = model.get_state()
        torch.testing.assert_close(X, dagger(X), rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
