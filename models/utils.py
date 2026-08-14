"""Shared helper routines for matrix models."""

from __future__ import annotations

import numpy as np
import torch

from matrix_hmc_track import config
from matrix_hmc_track.algebra import kron_2d


def parse_source(
    source: "np.ndarray | None",
    nmat: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor | None":
    """Convert a NumPy source array to an ``(nmat, N, N)`` configuration tensor.

    The *source* term is an external field coupled linearly to the matrices.
    Two input shapes are accepted:

    * ``(N,)`` — diagonal coupling to ``X[0]`` only.  The result has
      ``out[0] = diag(source)`` and ``out[1:] = 0``.
    * ``(nmat, N, N)`` — full coupling tensor applied to all matrices.

    Args:
        source: Source array or ``None`` (no source).
        nmat: Number of matrices in the model.
        device: Target PyTorch device.
        dtype: Target PyTorch dtype.

    Returns:
        Tensor of shape ``(nmat, N, N)`` on *device*, or ``None`` when
        *source* is ``None``.

    Raises:
        ValueError: If *source* has an incompatible shape.
    """
    if source is None:
        return None
    s = np.asarray(source)
    if s.ndim == 1:
        N = s.shape[0]
        out = torch.zeros(nmat, N, N, device=device, dtype=dtype)
        out[0] = torch.diag(torch.tensor(s, device=device, dtype=dtype))
        return out
    if s.ndim == 3:
        if s.shape[0] != nmat:
            raise ValueError(f"source has {s.shape[0]} matrices but model has nmat={nmat}")
        return torch.tensor(s, device=device, dtype=dtype)
    raise ValueError(f"source must be shape (N,) or (nmat,N,N), got {s.shape}")


def source_grad_inplace(source: torch.Tensor, grad: list, ncol: int, g: float) -> None:
    """Add the external-source contribution to the analytic gradient in-place.

    The linear source term in the action is ``-(N/sqrt(g)) Tr(J X)`` so its
    contribution to the gradient with respect to ``X[i]`` is
    ``-(N/sqrt(g)) * source[i]``.

    Args:
        source: Source tensor of shape ``(nmat, N, N)``.
        grad: List of ``(N, N)`` gradient tensors, one per matrix (modified
            in-place).
        ncol: Matrix size ``N``.
        g: Coupling constant ``g``.
    """
    coeff = -(ncol / g ** 0.5)
    for i in range(source.shape[0]):
        grad[i] = grad[i] + coeff * source[i]


def _commutator_action_sum(X: torch.Tensor) -> torch.Tensor:
    nmat = X.shape[0]
    if nmat < 2:
        return X.new_zeros((), dtype=X.dtype)

    total = torch.tensor(0.0, dtype=config.real_dtype, device=X.device)
    for i in range(nmat - 1):
        for j in range(i + 1, nmat):
            comm = X[i] @ X[j] - X[j] @ X[i]
            total = total + torch.trace(comm @ comm).real
    return total.to(dtype=X.dtype)


def _anticommutator_action_sum(X: torch.Tensor) -> torch.Tensor:
    nmat = X.shape[0]
    if nmat < 2:
        return X.new_zeros((), dtype=X.dtype)

    total = torch.tensor(0.0, dtype=config.real_dtype, device=X.device)
    for i in range(nmat - 1):
        for j in range(i + 1, nmat):
            anti = X[i] @ X[j] + X[j] @ X[i]
            total = total + torch.trace(anti @ anti).real
    return total.to(dtype=X.dtype)


def _fermion_det_log_identity_plus_sum_adX(X: torch.Tensor) -> torch.Tensor:
    """Return log|det(1 + i*sum_i ad_{X_i})| = sum_{a<b} log(1 + (mu_a - mu_b)^2).

    The i factor in the Yukawa term i*psibar*[sqrt(X^2), psi] means each eigenvalue
    of the fermion operator is (1 + i*delta), so |1 + i*delta|^2 = 1 + delta^2.
    Pairing (a,b) with (b,a) gives log(1 + delta^2) per pair -- always positive,
    no singularities, gradient 2*delta/(1+delta^2) bounded by 1.
    """
    sum_X2 = (X @ X).sum(dim=0)
    eigvals = torch.sqrt(torch.linalg.eigvalsh(sum_X2).real.to(dtype=config.real_dtype))
    N = eigvals.shape[0]
    i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=eigvals.device)
    delta = eigvals[j_idx] - eigvals[i_idx]   # >= 0 since eigvalsh returns sorted ascending
    return torch.log(1.0 + delta * delta).sum()


def gammaMajorana() -> torch.Tensor:
    """Construct the 4D Majorana gamma matrices ``Γ_1, ..., Γ_4`` and the charge-conjugation matrix.

    Uses the Majorana representation defined via Pauli matrices
    ``σ_1, σ_2, σ_3`` and the ``2×2`` identity ``Id``.  The resulting
    matrices satisfy the Clifford algebra
    ``{Γ_μ, Γ_ν} = 2 δ_{μν}``.

    Returns:
        Tuple ``(gammas, conj)`` where *gammas* is a tensor of shape
        ``(4, 4, 4)`` containing ``[Γ_1, Γ_2, Γ_3, Γ_4]`` and *conj* is the
        ``4×4`` charge-conjugation matrix, all on ``config.device`` with
        ``config.dtype``.
    """
    sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=config.dtype, device=config.device)
    sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=config.dtype, device=config.device)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=config.dtype, device=config.device)
    Id2 = torch.eye(2, dtype=config.dtype, device=config.device)
    gam0 = 1j * kron_2d(sigma2, Id2)
    gam1 = kron_2d(sigma3, Id2)
    gam2 = kron_2d(sigma1, sigma3)
    gam3 = kron_2d(sigma1, sigma1)
    gam4 = 1j * gam0
    conj = gam4
    gammas = torch.stack([gam1, gam2, gam3, gam4], dim=0)
    return gammas, conj


def gammaWeyl() -> torch.Tensor:
    """Construct the 4D Weyl-basis Dirac matrices ``Γ_1, ..., Γ_4``.

    The matrices are ``4×4`` block-off-diagonal in the Weyl (chiral) basis and
    satisfy the Clifford algebra ``{Γ_μ, Γ_ν} = 2 δ_{μν}``.

    Returns:
        Tensor of shape ``(4, 4, 4)`` containing ``[Γ_1, Γ_2, Γ_3, Γ_4]``
        on ``config.device`` with ``config.dtype``.
    """
    sigma1 = torch.tensor([[0, 1], [1, 0]], dtype=config.dtype, device=config.device)
    sigma2 = torch.tensor([[0, -1j], [1j, 0]], dtype=config.dtype, device=config.device)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=config.dtype, device=config.device)
    Id2 = torch.eye(2, dtype=config.dtype, device=config.device)

    gamma0 = -Id2
    gammas = (sigma1, sigma2, sigma3, 1j * gamma0)
    gamma_bars = gammas[:3] + (1j * (-gamma0),)

    zero2 = torch.zeros_like(Id2)

    def block(g, gb):
        return torch.cat((torch.cat((zero2, g), dim=1), torch.cat((gb, zero2), dim=1)), dim=0)

    return torch.stack([block(g, gb) for g, gb in zip(gammas, gamma_bars)], dim=0)

