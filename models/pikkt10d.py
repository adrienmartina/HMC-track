"""10D polarized IKKT model with fermionic Pfaffian."""

from __future__ import annotations

import os

import numpy as np
import torch

from matrix_hmc_track import config
from matrix_hmc_track.algebra import ad_matrix, ad_matrix_real_antisymmetric, get_trace_diag_indices_cached, random_hermitian, spinJMatrices
from matrix_hmc_track.models.base import MatrixModel
from matrix_hmc_track.models.utils import _commutator_action_sum, parse_source
from matrix_hmc_track.pfaffian import pfaffian, slogpfaff

model_name = "pikkt10d"


def build_model(args):
    return PIKKT10DModel(
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        massless=getattr(args, "massless", False),
        pfaffian_every=getattr(args, "pfaffian_every", 1),
        spin=getattr(args, "spin", None),
    )


class PIKKT10DModel(MatrixModel):
    """10D polarized IKKT model with ``Ω = 1`` and a full fermionic Pfaffian.

    Uses 10 bosonic ``N×N`` Hermitian traceless matrices.  The fermionic
    contribution is the Pfaffian of the 16N²-dimensional Dirac operator built
    from the 10D Gamma matrices.  The Pfaffian is computed via
    :func:`~matrix_hmc_track.pfaffian.slogpfaff` and is evaluated every
    *pfaffian_every* HMC steps for efficiency.

    A fuzzy-sphere (Myers-term) initial configuration can be loaded by passing
    a half-integer or integer *spin* value.

    Args:
        ncol: Matrix size ``N``.
        couplings: List of couplings; ``couplings[0]`` is the 't Hooft
            coupling ``g``.
        source: Optional external source (see
            :func:`~matrix_hmc_track.models.utils.parse_source`).
        massless: If ``True``, drop the mass (quadratic) terms. Default ``False``.
        pfaffian_every: Recompute the Pfaffian contribution to the potential
            every this many HMC trajectories. Default ``1`` (every step).
        spin: If not ``None``, initialise the first three matrices to a
            fuzzy-sphere background built from spin-``j`` matrices.
    """

    model_name = model_name

    def __init__(
        self,
        ncol: int,
        couplings: list,
        source: np.ndarray | None = None,
        massless: bool = False,
        pfaffian_every: int = 1,
        spin: float | None = None,
    ) -> None:
        super().__init__(nmat=10, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.omega = 1.0
        self.massless = massless
        self.pfaffian_every = int(pfaffian_every)
        self.spin = spin
        self._measure_calls = 0
        self.source = parse_source(source, self.nmat, config.device, config.dtype)
        self.is_hermitian = True
        self.is_traceless = True

        coeffs = torch.full((self.nmat,), 1.0 / 64.0, dtype=config.real_dtype, device=config.device)
        coeffs[:3] = 3.0 / 64.0
        self._mass_coeffs = coeffs

        # Precompute gamma_bar^I (10, 16, 16) and G^I = gamma_bar^{123} @ gamma_bar^I (10, 16, 16)
        self._gb, self._gb123, self._G = self._build_gammas(config.device, config.dtype)

    @staticmethod
    def _build_gammas(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build gamma_bar^I matrices and G^I = gamma_bar^{123} @ gamma_bar^I for I=1,...,10."""
        s1 = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
        s3 = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
        Id = torch.eye(2, dtype=dtype, device=device)

        def k4(a, b, c, d):
            return torch.kron(torch.kron(torch.kron(a, b), c), d)

        # gamma_bar^I (16x16) for I=1,...,10 (0-indexed)
        # For I=1,...,9: gamma_bar^I = gamma^I (lower-left block of Gamma^I)
        # For I=10: gamma_bar^{10} = -gamma^{10} = i * Id_16
        gb = torch.stack([
            k4(s2, s2, s1, Id),                           # I=1
            k4(s2, s3, Id, s2),                           # I=2
            -k4(s3, Id, Id, Id),                          # I=3
            -k4(s2, s1, Id, s2),                          # I=4
            k4(s2, Id, s2, s3),                           # I=5
            k4(s1, Id, Id, Id),                           # I=6
            k4(s2, s2, s2, s2),                           # I=7
            -k4(s2, Id, s2, s1),                          # I=8
            k4(s2, s2, s3, Id),                           # I=9
            1j * torch.eye(16, dtype=dtype, device=device),  # I=10
        ], dim=0)  # (10, 16, 16)

        # gamma_bar^{123} = (1/4) * {gamma_bar^1, [gamma_bar^2, gamma_bar^3]}
        g1, g2, g3 = gb[0], gb[1], gb[2]
        comm23 = g2 @ g3 - g3 @ g2
        gb123 = 0.25 * (g1 @ comm23 + comm23 @ g1)

        # G^I = gamma_bar^{123} @ gamma_bar^I
        G = torch.einsum("ab,Ibc->Iac", gb123, gb)  # (10, 16, 16)

        return gb, gb123, G

    def load_fresh(self):
        scale = float(np.sqrt(self.g / self.ncol))
        X = scale * random_hermitian(self.ncol, batchsize=self.nmat)

        if self.spin is not None:
            J_matrices = torch.from_numpy(spinJMatrices(self.spin)).to(
                dtype=config.dtype, device=config.device
            )
            ntimes = self.ncol // J_matrices.shape[1]
            eye_nt = torch.eye(ntimes, dtype=config.dtype, device=config.device)
            dim = ntimes * J_matrices.shape[1]

            X = torch.zeros_like(X)
            for i in range(3):
                X[i][:dim, :dim] = 3/8 * torch.kron(eye_nt, J_matrices[i])

        self.set_state(X)

    def fermion_determinant(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Compute ``-0.5 * log|det M_F(X)|`` for the fermionic contribution to the action.

        The fermion matrix ``M_F`` is constructed in the ``(16 N²)``-dimensional
        adjoint representation using the precomputed 10D Gamma matrices.  Zero
        modes from the trace sector are lifted by adding a rank-1 projector to
        each diagonal ``N²×N²`` block before taking the log-determinant.

        Args:
            X: Configuration tensor of shape ``(10, N, N)``.  Defaults to the
                current internal state.

        Returns:
            Scalar real-valued tensor equal to ``-0.5 * log|det M_F|``.
        """
        X = self._resolve_X(X)
        N2 = self.ncol ** 2

        N = self.ncol
        adX = 1j * ad_matrix(X)  # (10, N^2, N^2) — batched over all 10 matrices at once

        if self.massless:
            # Pure IKKT: Pf(i/2 * gamma_bar^I ⊗ ad_{X_I})
            # contribution = -0.5 * log|det(i/2 * sum_I gamma_bar^I ⊗ ad_{X_I})|
            gb = self._gb.to(device=X.device, dtype=X.dtype)
            # sum_I kron(gb[I], adX[I]): result[i*N2+k, j*N2+l] = sum_I gb[I,i,j]*adX[I,k,l]
            M = 0.5j * torch.einsum("Iij,Ikl->ikjl", gb, adX).reshape(16 * N2, 16 * N2)
        else:
            # pIKKT: Pf(i/2 * gamma_bar^I ⊗ ad_{X_I} + i/8 * gamma_bar^{123} ⊗ 1)
            # after factoring: -0.5 * log|det(1 - 4 G^I ⊗ ad_{X_I})|
            G = self._G.to(device=X.device, dtype=X.dtype)
            kron_sum = torch.einsum("Iij,Ikl->ikjl", G, adX).reshape(16 * N2, 16 * N2)
            M = torch.eye(16 * N2, dtype=X.dtype, device=X.device) - 4.0 * kron_sum

        # Lift the trace-mode zero in all 16 diagonal N^2 x N^2 blocks simultaneously.
        # Adds (1/N)|vec(I)><vec(I)| to M[j*N2+d_i, j*N2+d_j] for all j, d_i, d_j.
        diag_idx = get_trace_diag_indices_cached(N, M.device)           # (N,)
        offsets = torch.arange(16, dtype=torch.long, device=M.device) * N2  # (16,)
        g = (offsets.unsqueeze(1) + diag_idx.unsqueeze(0))             # (16, N)
        M[g.unsqueeze(2).expand(16, N, N).reshape(-1),
          g.unsqueeze(1).expand(16, N, N).reshape(-1)] += 1.0 / N

        _, logdet = torch.slogdet(M)
        return -0.5 * logdet
        # slogdet = torch.slogdet(M)
        # return torch.tensor([slogdet[0], -0.5 * slogdet[1]])
    
    def fermion_pfaffian(self, X: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sign, log|pf(M_F)|)`` for the full fermionic Pfaffian.

        Delegates to :func:`~matrix_hmc_track.pfaffian.slogpfaff` on the
        ``(16(N²-1))``-dimensional real antisymmetric fermion matrix built via
        :meth:`fermion_matrix`.

        Args:
            X: Configuration tensor of shape ``(10, N, N)``.  Defaults to the
                current internal state.

        Returns:
            Tuple ``(sign, log_abs)`` — see :func:`~matrix_hmc_track.pfaffian.slogpfaff`.
        """
        M = self.fermion_matrix(X)
        return slogpfaff(M)

    def fermion_matrix(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Build the ``(16(N²-1))``-dimensional skew-symmetric fermion matrix.

        Constructs ``M_F`` in the real antisymmetric basis of ``su(N)`` using
        :func:`~matrix_hmc_track.algebra.ad_matrix_real_antisymmetric`.

        * **Massless**: ``M = (i/2) sum_I gamma_bar^I ⊗ ad_{X_I}``.
        * **Massive** (pIKKT): adds a constant block
          ``(i/8) gamma_bar^{123} ⊗ 1``.

        Args:
            X: Configuration tensor of shape ``(10, N, N)``.  Defaults to the
                current internal state.

        Returns:
            Skew-symmetric tensor of shape ``(16(N²-1), 16(N²-1))``.
        """
        X = self._resolve_X(X)
        N2 = self.ncol ** 2 - 1

        N = self.ncol
        adX = ad_matrix_real_antisymmetric(X, traceless=self.is_traceless).to(dtype=X.dtype)  # (10, N^2-1, N^2-1)

        if self.massless:
            gb = self._gb.to(device=X.device, dtype=X.dtype)
            M = 0.5j * torch.einsum("Iij,Ikl->ikjl", gb, adX).reshape(16 * N2, 16 * N2)
        else:
            gb, gb123 = self._gb.to(device=X.device, dtype=X.dtype), self._gb123.to(device=X.device, dtype=X.dtype)
            kron_sum = torch.einsum("Iij,Ikl->ikjl", gb, adX).reshape(16 * N2, 16 * N2)
            M = 0.5j * kron_sum + 0.125j * torch.kron(gb123, torch.eye(N2, dtype=X.dtype, device=X.device))

        # pfaffian = slogpfaff(M)
        return M

    def bosonic_potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the purely bosonic part of the potential ``V_bos(X)``.

        Combines the Yang-Mills commutator term with anisotropic mass terms
        (weighted by precomputed coefficients) and, if not massless, a
        Myers/cubic term ``(i/3) ε^{ijk} Tr(X_i X_j X_k)`` for ``i,j,k ∈ {1,2,3}``.

        Args:
            X: Configuration tensor of shape ``(10, N, N)``.  Defaults to the
                current internal state.

        Returns:
            Scalar real-valued tensor ``(N/g) * V_bos``.
        """
        X = self._resolve_X(X)

        bos = -0.5 * _commutator_action_sum(X)

        if not self.massless:
            trace_sq = torch.einsum("bij,bji->b", X, X).real
            bos = bos + torch.dot(self._mass_coeffs.to(device=X.device), trace_sq)
            # (i/3) * eps^{ijk} Tr(X_i X_j X_k) with i,j,k in {1,2,3}
            myers = 1j * (torch.trace(X[0] @ X[1] @ X[2]) - torch.trace(X[0] @ X[2] @ X[1]))
            bos = bos + myers

        return (self.ncol / self.g) * bos.real

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        src = torch.tensor(0.0, dtype=config.real_dtype, device=X.device)
        if self.source is not None:
            src = (-(self.ncol / self.g ** 0.5) * torch.einsum("iab,iba->", self.source, X)).real
        return self.bosonic_potential(X) + self.fermion_determinant(X) + src

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            self._measure_calls += 1
            eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X]
            eigs.append(
                torch.linalg.eigvalsh(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2])
                .cpu()
                .numpy()
            )
            if (self._measure_calls % self.pfaffian_every) == 0:
                pf0 = self.fermion_pfaffian(X)[0]
            else:
                pf0 = torch.full((), complex(float("nan"), float("nan")), dtype=X.dtype, device=X.device)

            trace_sq = torch.einsum("bij,bji->b", X, X).real
            tr_i = trace_sq[:3].sum() / (3 * self.ncol)
            tr_p = trace_sq[3:].sum() / (7 * self.ncol)
            comm = _commutator_action_sum(X).real / self.ncol
            corrs = torch.stack([tr_i, tr_p, comm, pf0]).cpu().numpy()

        return eigs, corrs

    def build_paths(
        self,
        name_prefix: str,
        data_path: str,
        *,
        step_size: float | None = None,
        nsteps: int | None = None,
        niters: int | None = None,
    ) -> dict[str, str]:
        variant = "_ikkt" if self.massless else ""
        hmc_suffix = self._hmc_path_suffix(step_size=step_size, nsteps=nsteps, niters=niters)
        run_dir = os.path.join(
            data_path,
            (
                f"{name_prefix}_{self._path_model_name()}{variant}_"
                f"g{round(self.g, 4)}_N{self.ncol}{hmc_suffix}"
            ),
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        if self.massless:
            fdet_str = "-0.5 log|det(i/2 gamma_bar^I x ad_XI)|  [IKKT]"
        else:
            fdet_str = "-0.5 log|det(1 - 4 G^I x ad_XI)|  [pIKKT]"
        lines = [
            f"  Coupling g               = {self.g}",
            f"  Fermion determinant      = {fdet_str}",
            f"  Pfaffian every           = {self.pfaffian_every}",
        ]
        if not self.massless:
            lines.insert(1, "  Omega                    = 1 (fixed)")
        return lines

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        trace_sq = torch.einsum("bij,bji->b", X, X).real
        tr_i = (trace_sq[:3].sum() / (3 * self.ncol)).item()
        tr_p = (trace_sq[3:].sum() / (7 * self.ncol)).item()
        return f"<tr Xi^2> = {tr_i:.5f}, <tr Xp^2> = {tr_p:.5f}. "

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "pikkt10d",
                "omega_fixed": 1.0,
                "fermion_determinant": "ikkt_pfaffian" if self.massless else "pikkt_pfaffian",
                "massless": self.massless,
                "pfaffian_every": self.pfaffian_every,
            }
        )
        return meta
