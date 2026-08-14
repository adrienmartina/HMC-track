"""Type I polarized IKKT model."""

from __future__ import annotations

import os

import numpy as np
import torch

from matrix_hmc_track import config
from matrix_hmc_track.algebra import (
    add_trace_projector_inplace,
    ad_matrix,
    get_eye_cached,
    random_hermitian,
)
from matrix_hmc_track.models.base import MatrixModel
from matrix_hmc_track.models.utils import _commutator_action_sum, parse_source
model_name = "pikkt4d_type1"


def build_model(args):
    return PIKKTTypeIModel(
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        eta=getattr(args, "eta", 1.0),
        massless=getattr(args, "massless", False),
    )


def _type1_logdet_impl(X: torch.Tensor, A: torch.Tensor, eta: float = 1.0, massless: bool = False) -> torch.Tensor:
    """Original implementation — kept for benchmarking against _type1_logdet_impl."""
    # eta rescales the constant fermion block: A -> eta*A and A^{-1} -> (1/eta)*A^{-1}.
    eta_t = torch.tensor(eta, dtype=X.dtype, device=X.device)
    adX = 1j * ad_matrix(X[:4])
    adX1, adX2, adX3, adX4 = adX
    i = torch.tensor(1j, dtype=X.dtype, device=X.device)
    twoi = torch.tensor(2j, dtype=X.dtype, device=X.device)

    upper_left = -(adX3 + i * adX4)
    upper_right = -(adX1 - i * adX2)
    lower_left = -(adX1 + i * adX2)
    lower_right = adX3 - i * adX4

    C = torch.cat(
        (torch.cat((upper_left, lower_left), dim=1), torch.cat((upper_right, lower_right), dim=1)),
        dim=0,
    )
    N = X.shape[-1]
    dim = N * N

    if massless:
        B = torch.cat(
        (torch.cat((upper_left, upper_right), dim=1), torch.cat((lower_left, lower_right), dim=1)),
        dim=0,
        )
        add_trace_projector_inplace(C[:dim, :dim], N)
        add_trace_projector_inplace(C[dim:, dim:], N)
        add_trace_projector_inplace(B[:dim, :dim], N)
        add_trace_projector_inplace(B[dim:, dim:], N)
        det = torch.slogdet(C) + torch.slogdet(B)
    else:
        AB = torch.cat(
            (
                torch.cat((twoi * lower_left, twoi * lower_right), dim=1),
                torch.cat((-twoi * upper_left, -twoi * upper_right), dim=1),
            ),
            dim=0,
        )

        K = -eta_t * A - (0.25 / eta_t) * (C @ AB)

        # Lift zero modes from the trace sector (identity direction)
        # This adds a constant mass term to the trace mode, ensuring invertibility
        # without affecting the physics (since it's a constant factor in det).
        add_trace_projector_inplace(K[:dim, :dim], N)
        add_trace_projector_inplace(K[dim:, dim:], N)

        det = torch.slogdet(K)

    return det

def _type1_massless_staudacher(X: torch.Tensor) -> torch.Tensor:
    """Using Krauth-Staudacher formula for the massless case, as described in their 1998 paper (https://arxiv.org/abs/hep-th/9803117). Kept for benchmarking.
    det ((adX4+i adX3, i adX2 + adX1)
         (i adX2-adX1, adX4 - i adX3))
    """

    adX1, adX2, adX3, adX4 = 1j * ad_matrix(X[:4])

    mat = torch.cat(
        (
            torch.cat((adX4 + 1j * adX3, 1j * adX2 + adX1), dim=1),
            torch.cat((1j * adX2 - adX1, adX4 - 1j * adX3), dim=1),
        ),
        dim=0,
    )

    add_trace_projector_inplace(mat[:mat.shape[0] // 2, :mat.shape[1] // 2], X.shape[-1])
    add_trace_projector_inplace(mat[mat.shape[0] // 2:, mat.shape[1] // 2:], X.shape[-1])

    det = torch.slogdet(mat)

    return det


class PIKKTTypeIModel(MatrixModel):
    """Type I polarized IKKT model with 4 bosonic matrices and a fermionic determinant.

    The action consists of a bosonic Yang-Mills term and the logarithm of a
    fermionic determinant derived from the Type-I superstring matrix model:

    .. math::

        S = \\frac{N}{g} \\left[ \\mathrm{Tr}(X_i^2)
            - \\frac{1}{2} \\sum_{i<j} \\mathrm{Tr}([X_i, X_j]^2) \\right]
            - \\frac{1}{2} \\log |\\det M_F(X)|

    where ``M_F`` encodes the adjoint-action Dirac operator for 4 matrices and
    the ``\\eta`` parameter controls the constant (mass) sector.

    Args:
        ncol: Matrix size ``N``.
        couplings: List of couplings; ``couplings[0]`` is the 't Hooft
            coupling ``g``.
        source: Optional external source (see
            :func:`~matrix_hmc_track.models.utils.parse_source`).
        eta: Rescaling factor for the constant fermion block. Default ``1.0``.
        massless: If ``True``, use the massless Krauth-Staudacher formula
            (zero quadratic term). Default ``False``.
    """

    model_name = model_name

    def __init__(self, ncol: int, couplings: list, source: np.ndarray | None = None, eta: float = 1.0, massless: bool = False) -> None:
        super().__init__(nmat=4, ncol=ncol)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.source = parse_source(source, self.nmat, config.device, config.dtype)
        self.eta = float(eta)
        self.massless = massless
        self.is_hermitian = True
        self.is_traceless = True

        dim_tr = self.ncol * self.ncol
        eye_tr = get_eye_cached(dim_tr, device=config.device, dtype=config.dtype)
        i = 1j
        two_i_I = (2 * i) * eye_tr
        A = torch.zeros((2 * dim_tr, 2 * dim_tr), device=config.device, dtype=config.dtype)
        A[:dim_tr, dim_tr:] = two_i_I
        A[dim_tr:, :dim_tr] = -two_i_I
        self._type1_A = A.clone()

        def base_fn(X: torch.Tensor, *, model=self) -> torch.Tensor:
            if model.massless:
                return 2 * _type1_massless_staudacher(X)
            return _type1_logdet_impl(X, model._type1_A, eta=model.eta, massless=model.massless)

        self._log_det_fn = base_fn

    def load_fresh(self):
        X = 0.01 * random_hermitian(self.ncol, batchsize=self.nmat)
        self.set_state(X)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        bos = -0.5 * _commutator_action_sum(X)
        trace_sq = 0.0 if self.massless else torch.einsum("bij,bji->", X, X)
        bos = bos + trace_sq

        det = -0.5 * self._log_det_fn(X)[1].real
        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / self.g ** 0.5) * torch.einsum("iab,iba->", self.source, X)

        return (bos.real * (self.ncol / self.g)) + det + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = []
            for i in range(self.nmat):
                e = torch.linalg.eigvalsh(X[i]).cpu().numpy()
                eigs.append(e)

            e_complex = torch.linalg.eigvals((X[0] + 1j * X[1])).cpu().numpy()
            eigs.append(e_complex)
            e_complex = torch.linalg.eigvals((X[2] + 1j * X[3])).cpu().numpy()
            eigs.append(e_complex)

            eigs.append(torch.linalg.eigvalsh(X[0] @ X[0] + X[1] @ X[1] + X[2] @ X[2]).cpu().numpy())

            C = X[0] @ X[1] - X[1] @ X[0]
            A = X[0] @ X[1] + X[1] @ X[0]
            c1 = torch.trace(C @ C).real
            c2 = torch.trace(A @ A).real
            C = X[2] @ X[3] - X[3] @ X[2]
            A = X[2] @ X[3] + X[3] @ X[2]
            c3 = torch.trace(C @ C).real
            c4 = torch.trace(A @ A).real

            corrs = torch.stack([c1, c2, c3, c4]).cpu().numpy()

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
        eta_suffix = f"_eta{round(self.eta, 4)}" if self.eta != 1.0 else ""
        hmc_suffix = self._hmc_path_suffix(step_size=step_size, nsteps=nsteps, niters=niters)
        run_dir = os.path.join(
            data_path,
            (
                f"{name_prefix}_{self._path_model_name()}_g{round(self.g, 4)}"
                f"{eta_suffix}_N{self.ncol}{hmc_suffix}"
            ),
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        trX1 = (torch.trace(X[0] @ X[0]) / self.ncol).item().real
        trX4 = (torch.trace(X[3] @ X[3]) / self.ncol).item().real
        return f"trX_1^2 = {trX1:.5f}, trX_4^2 = {trX4:.5f}. "

    def extra_config_lines(self) -> list[str]:
        return [
            f"  Coupling g               = {self.g}",
            f"  SUSY breaking parameter eta = {self.eta}",
        ]

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "type1",
                "eta": self.eta,
            }
        )
        if self.massless:
            meta["massless"] = True
        return meta
