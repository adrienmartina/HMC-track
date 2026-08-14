"""Adjoint determinant matrix model."""

from __future__ import annotations

import os

import numpy as np
import torch

from matrix_hmc_track import config
from matrix_hmc_track.models.base import MatrixModel
from matrix_hmc_track.models.utils import (
    _anticommutator_action_sum,
    _commutator_action_sum,
    _fermion_det_log_identity_plus_sum_adX,
    parse_source,
)
from matrix_hmc_track.algebra import random_hermitian
model_name = "adjoint_det"

def build_model(args):
    if args.nmat is None:
        raise ValueError("--nmat must be provided for adjoint_det model")
    return AdjointDetModel(
        dim=args.nmat,
        ncol=args.ncol,
        couplings=args.coupling,
        source=args.source,
        det_coeff=args.det_coeff,
    )


class AdjointDetModel(MatrixModel):
    r"""Matrix model whose fermion sector is ``det(1 + i \sum_i \mathrm{ad}_{\sqrt{X^2}})``.

    The action combines a Yang-Mills bosonic term with a simplified fermionic
    determinant that avoids the full Pfaffian computation:

    .. math::

        S = \frac{N}{g} \left[ \mathrm{Tr}({\sum_i X_i^2})
            - \frac{1}{2} \sum_{i<j} \mathrm{Tr}([X_i, X_j]^2) \right]
            - c \cdot (D-2) \sum_{a<b} \log\!\left(1 + (\mu_a - \mu_b)^2\right)

    where ``\mu_a`` are the eigenvalues of ``\sqrt{\sum_i X_i^2}``, ``D = nmat``
    is the dimension, and ``c`` is *det_coeff*.

    Args:
        dim: Number of matrices ``D``.
        ncol: Matrix size ``N``.
        couplings: List of couplings; ``couplings[0]`` is ``g``.
        source: Optional external source (see
            :func:`~matrix_hmc_track.models.utils.parse_source`).
        det_coeff: Coefficient *c* in front of the fermionic log-det.
            Default ``0.5``.
    """

    model_name = model_name

    def __init__(self, dim: int, ncol: int, couplings: list, source: np.ndarray | None = None, det_coeff: float = 0.5) -> None:
        super().__init__(nmat=dim, ncol=ncol)
        self.source = parse_source(source, dim, config.device, config.dtype)
        self.couplings = couplings
        self.g = self.couplings[0]
        self.det_coeff = det_coeff
        self.is_hermitian = True
        self.is_traceless = True

    def load_fresh(self):
        # X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        X = 0.01 * random_hermitian(self.ncol, traceless=self.is_traceless, batchsize=self.nmat)
        if self.source is not None:
            X = self.source / 2
        self.set_state(X)

    def _fermion_det(self, X: torch.Tensor) -> torch.Tensor:
        r"""Return log|det(1 + i sum_i ad_{\sqrt{X^2}})|."""
        sum_X2 = (X @ X).sum(dim=0)
        eigvals = torch.sqrt(torch.linalg.eigvalsh(sum_X2).real.to(dtype=config.real_dtype))
        N = eigvals.shape[0]
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=eigvals.device)
        delta = eigvals[j_idx] - eigvals[i_idx]   # >= 0 since eigvalsh returns sorted ascending
        return torch.log(1 + delta * delta).sum()

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        sum_X2 = (X @ X).sum(dim=0)
        trace_sq = torch.einsum("ii->", sum_X2).real
        # quartic = torch.einsum("ii->", sum_X2 @ sum_X2).real
        comm_term = -0.5 * _commutator_action_sum(X).real
        bos = trace_sq + comm_term

        det = -self.det_coeff * (self.nmat - 2) * self._fermion_det(X)

        src = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        if self.source is not None:
            src = -(self.ncol / self.g ** 0.5) * torch.einsum("iab,iba->", self.source, X)

        return (self.ncol / self.g) * bos + det + src.real

    def measure_observables(self, X: torch.Tensor | None = None):
        with torch.no_grad():
            X = self._resolve_X(X)
            eigs = [torch.linalg.eigvalsh(mat).cpu().numpy() for mat in X]
            eigs.append(torch.linalg.eigvals(X[0] + 1j * X[1]).cpu().numpy())
            comm_raw = _commutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol
            anticomm_raw = _anticommutator_action_sum(X).real.item() / self.nmat / (self.nmat - 1) / self.ncol

            # Moments tr(X_i X_j) to diagnose emergent rotational symmetry.
            moments = torch.einsum("aij,bji->ab", X, X).real
            corrs = np.concatenate(
                (
                    np.array([anticomm_raw, comm_raw], dtype=np.float64),
                    moments.detach().cpu().numpy().astype(np.float64).reshape(-1),
                )
            )
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
        hmc_suffix = self._hmc_path_suffix(step_size=step_size, nsteps=nsteps, niters=niters)
        run_dir = os.path.join(
            data_path,
            (
                f"{name_prefix}_{self._path_model_name()}_D{self.nmat}_"
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
        return [f"  Coupling g               = {self.g}", f"  Dimension D             = {self.nmat}"]

    def status_string(self, X: torch.Tensor | None = None) -> str:
        X = self._resolve_X(X)
        # return f"tr X_1^2 = {torch.trace(X[0] @ X[0]).real.item() / self.ncol:.5f} , tr X_{self.nmat}^2 = {torch.trace(X[-1] @ X[-1]).real.item() / self.ncol:.5f}"
        return 'tr X_i^2 = ' + ','.join([f"{torch.trace(X[i] @ X[i]).real.item() / self.ncol / np.sqrt(self.g):.2f}" for i in range(self.nmat)])

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "has_source": self.source is not None,
                "model_variant": "adjoint_det",
            }
        )
        return meta
