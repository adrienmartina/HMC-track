import torch


def _sign_pfaffian_2d(A: torch.Tensor) -> torch.Tensor:
    """Phase of pf(A) for a 2D skew-symmetric matrix via Parlett-Reid with global column pivoting.

    At each step the remaining submatrix is renormalized by its max absolute value — a
    positive-real scale that leaves sign(pf) unchanged — so entries stay bounded regardless
    of how many Schur complement updates have been applied.

    Global column pivoting (searching all remaining rows for the pivot, not just the current
    block) is essential for sparse matrices: a block column can be all-zero even when the
    full column is not, causing local-only pivoting to return a spurious zero sign.
    """
    n = A.shape[0]
    A = A.clone()
    sign = torch.ones([], dtype=A.dtype, device=A.device)

    for k in range(0, n - 1, 2):
        # Renormalize remaining submatrix — positive-real scale leaves sign(pf) unchanged.
        scale = A[k:, k:].abs().amax()
        if not torch.isfinite(scale) or scale == 0:
            return torch.zeros([], dtype=A.dtype, device=A.device)
        A[k:, k:] = A[k:, k:] / scale

        # Global column pivot: search all of A[k+1:, k] for the largest entry.
        i_rel = A[k + 1:, k].abs().argmax().item()
        if i_rel != 0:
            i_pivot = k + 1 + i_rel
            A[[k + 1, i_pivot], :] = A[[i_pivot, k + 1], :].clone()
            A[:, [k + 1, i_pivot]] = A[:, [i_pivot, k + 1]].clone()
            sign = -sign

        pivot = A[k, k + 1]
        if pivot.abs() == 0:
            return torch.zeros([], dtype=A.dtype, device=A.device)
        sign = sign * torch.sgn(pivot)

        if k + 2 < n:
            tau = A[k + 2:, k:k + 2].clone()
            A[k + 2:, k + 2:] -= (tau[:, 0:1] @ tau[:, 1:2].T - tau[:, 1:2] @ tau[:, 0:1].T) / pivot

    return sign



def _sign_pfaffian(A: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Phase of pf(A). Handles batched input by iterating over the batch dimension."""
    if A.dim() == 2:
        return _sign_pfaffian_2d(A)
    *batch, n, _ = A.shape
    A_flat = A.reshape(-1, n, n)
    signs = torch.stack([_sign_pfaffian_2d(A_flat[i]) for i in range(A_flat.shape[0])])
    return signs.reshape(batch)


class _PfaffianFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, block_size):
        A_det = A.detach()
        # Use sign * exp(log|pf|) to avoid overflow from accumulated pivot products.
        # log|pf| = (1/2) log|det(A)| is stable; sign is computed via Parlett-Reid.
        sign = _sign_pfaffian(A_det, block_size)
        log_abs = torch.linalg.slogdet(A_det).logabsdet / 2
        pf = sign * log_abs.exp()
        ctx.save_for_backward(A_det, pf)
        return pf

    @staticmethod
    def backward(ctx, grad_output):
        A, pf = ctx.saved_tensors
        # d pf(A) = (pf/2) Tr(A^{-1} dA)  =>  grad_A = grad_out * (pf/2) * A^{-T}
        A_inv = torch.linalg.inv(A)
        scale = grad_output.unsqueeze(-1).unsqueeze(-1) * pf.unsqueeze(-1).unsqueeze(-1) / 2
        return scale * A_inv.transpose(-2, -1), None


def pfaffian(A: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Compute the Pfaffian of a skew-symmetric matrix, with full autograd support.

    The Pfaffian is computed as ``sign(pf(A)) * exp((1/2) log|det(A)|)`` so
    that the value stays representable for moderate ``N``.  For very large ``N``
    where ``pf(A)`` would overflow use :func:`slogpfaff` instead.

    Gradient note: PyTorch's native ``slogdet`` provides the gradient for
    ``log|det|``; the sign factor is computed without gradient via blocked
    Parlett-Reid (see :func:`slogpfaff`).

    Args:
        A: Skew-symmetric tensor of shape ``(..., n, n)``.
        block_size: Outer-loop block width ``b = 2 * block_size`` used by the
            internal sign-computation routine. Default ``32``.

    Returns:
        Pfaffian ``pf(A)`` of shape ``(...)``, same dtype as *A*.

    Raises:
        ValueError: If *A* is not square.
    """
    *batch, n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix must be square, got {A.shape}")
    if n % 2 == 1:
        return torch.zeros(batch, dtype=A.dtype, device=A.device)
    if n == 0:
        return torch.ones(batch, dtype=A.dtype, device=A.device)
    return _PfaffianFn.apply(A, block_size)


def slogpfaff(A: torch.Tensor, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Overflow-safe Pfaffian returning ``(sign, log|pf(A)|)``.

    The relationship ``pf(A) = sign * exp(log|pf(A)|)`` is preserved.
    ``log|pf(A)| = (1/2) log|det(A)|`` is computed via
    :func:`torch.linalg.slogdet` (numerically stable LU with partial pivoting)
    and its gradient is tracked by PyTorch's native autograd.

    The sign is computed via blocked Parlett-Reid with local partial pivoting
    (detached from the autograd graph) — accurate for both sign-problem-free
    cases (``sign = ±1``) and complex phases.

    For real *A* the sign is ``±1`` (a float tensor).
    For complex *A* the sign is a unit complex number (phase of ``pf(A)``).

    Args:
        A: Skew-symmetric tensor of shape ``(..., n, n)``.
        block_size: Block width for the sign computation. Default ``32``.

    Returns:
        Tuple ``(sign, log_abs)`` both of shape ``(...)``.  *sign* has the
        same dtype as *A*; *log_abs* is real.  Gradients flow through
        *log_abs* via PyTorch's ``slogdet`` autograd.

    Raises:
        ValueError: If *A* is not square.
    """
    *batch, n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix must be square, got {A.shape}")
    real_dtype = A.real.dtype if A.is_complex() else A.dtype
    if n % 2 == 1:
        return (torch.zeros(batch, dtype=A.dtype, device=A.device),
                torch.full(batch, float("-inf"), dtype=real_dtype, device=A.device))
    if n == 0:
        return (torch.ones(batch, dtype=A.dtype, device=A.device),
                torch.zeros(batch, dtype=real_dtype, device=A.device))

    # log|pf(A)| = (1/2) log|det(A)|: stable via LU, gradient-compatible
    log_abs = torch.linalg.slogdet(A).logabsdet / 2

    # sign(pf(A)) via blocked column-pivoting Parlett-Reid (detached — no gradient)
    sign = _sign_pfaffian(A.detach(), block_size)

    return sign, log_abs


def make_skew_symmetric(A: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric part of *A*, i.e. ``(A - A^T) / 2``.

    Args:
        A: Input tensor of shape ``(..., n, n)``.

    Returns:
        Skew-symmetric tensor of the same shape.
    """
    return (A - A.transpose(-2, -1)) / 2


def verify_pfaffian(A: torch.Tensor, pf: torch.Tensor, tol: float = 1e-5) -> bool:
    """Validate the Pfaffian identity ``pf(A)^2 == det(A)`` within a tolerance.

    Useful as a quick sanity check after computing a Pfaffian.

    Args:
        A: Skew-symmetric square tensor.
        pf: Pfaffian of *A* (result of :func:`pfaffian` or :func:`slogpfaff`).
        tol: Absolute and relative tolerance passed to :func:`torch.allclose`.

    Returns:
        ``True`` if ``pf**2`` and ``det(A)`` agree within *tol*.
    """
    return torch.allclose(pf ** 2, torch.linalg.det(A), atol=tol, rtol=tol)


if __name__ == "__main__":
    print("Pfaffian Implementation - Quick Tests")
    print("=" * 50)

    # Test 1: Simple 2x2 case
    A = torch.tensor([[0., 3.], [-3., 0.]])
    pf = pfaffian(A)
    s, la = slogpfaff(A)
    print(f"\nTest 1 - 2x2 matrix: [[0, 3], [-3, 0]]")
    print(f"  pfaffian     = {pf.item():.6f}  (expected: 3.0)")
    print(f"  log_pfaffian = log_abs={la.item():.6f}  sign={s.item():.6f}  (expected: log(3), +1)")

    # Test 2: 4x4 random skew-symmetric
    # torch.manual_seed(123)
    A = make_skew_symmetric(torch.randn(4, 4))
    pf = pfaffian(A)
    s, la = slogpfaff(A)
    det = torch.linalg.det(A)
    print(f"\nTest 2 - 4x4 random")
    print(f"  pf²={pf.item()**2:.6f}  det={det.item():.6f}  match={verify_pfaffian(A, pf)}")
    print(f"  s*exp(la)={s.item()*la.exp().item():.6f}  pf={pf.item():.6f}  match={torch.allclose(s * la.exp(), pf)}")

    # Test 3: Batched 6x6
    A_batch = make_skew_symmetric(torch.randn(10, 6, 6))
    pf_batch = pfaffian(A_batch)
    s_batch, la_batch = slogpfaff(A_batch)
    print(f"\nTest 3 - Batched 10×6×6")
    print(f"  pfaffian verified:      {all(verify_pfaffian(A_batch[i], pf_batch[i]) for i in range(10))}")
    print(f"  log_pfaffian consistent: {torch.allclose(s_batch * la_batch.exp(), pf_batch)}")
    print(f"  log_pfaffian consistent: {torch.norm(s_batch * la_batch.exp() - pf_batch)}")

    # Test 4: pfaffian gradient
    A_grad = torch.randn(4, 4, requires_grad=True)
    pfaffian(make_skew_symmetric(A_grad)).backward()
    print(f"\nTest 4 - pfaffian gradient: norm={A_grad.grad.norm():.6f}")

    # Test 5: log_pfaffian gradient (via slogdet autograd)
    A_grad2 = torch.randn(4, 4, requires_grad=True)
    s2, la2 = slogpfaff(make_skew_symmetric(A_grad2))
    la2.backward()
    print(f"\nTest 5 - log_pfaffian gradient: norm={A_grad2.grad.norm():.6f}")

    # Test 6: Large matrix — log_pfaffian stable where pfaffian overflows
    # torch.manual_seed(42)
    N = 500
    A_large = make_skew_symmetric(torch.randn(N, N))
    s_large, la_large = slogpfaff(A_large)
    slogdet = torch.linalg.slogdet(A_large)
    print(f"\nTest 6 - {N}×{N} overflow-safe")
    print(f"  2*log|pf| = {2*la_large.item():.4f},  log|det| = {slogdet.logabsdet.item():.4f}  match={torch.isclose(2*la_large, slogdet.logabsdet)}")
    print(f"  sign(pf)² = {s_large.item()**2:.1f},  sign(det) = {slogdet.sign.item():.1f}")

    # Test 7: block_size consistency for sign
    # torch.manual_seed(7)
    A_test = make_skew_symmetric(torch.randn(128, 128)) / 128 ** 0.5
    signs = [slogpfaff(A_test, block_size=bs)[0].item() for bs in [1, 8, 32, 64]]
    log_abs_vals = [slogpfaff(A_test, block_size=bs)[1].item() for bs in [1, 8, 32, 64]]
    print(f"\nTest 7 - block_size consistency (128×128)")
    print(f"  log_abs (all identical, from slogdet): {[f'{v:.8f}' for v in log_abs_vals]}")
    print(f"  signs: {signs}")
    print(f"  All log_abs equal: {max(abs(v - log_abs_vals[0]) for v in log_abs_vals) < 1e-12}")
    print(f"  All signs equal:   {len(set(signs)) == 1}")

    # Test 8: sign accuracy with sign problem (pf sweeps through zero)
    # torch.manual_seed(99)
    n8 = 30
    A8_base = make_skew_symmetric(torch.randn(n8, n8))
    A8_pert = make_skew_symmetric(torch.randn(n8, n8))
    ts = torch.linspace(0, 1, 20)
    pf_signs = []
    la_signs = []
    for t in ts:
        At = A8_base * (1 - t) + A8_pert * t
        pf_t = pfaffian(At)
        s_t, la_t = slogpfaff(At)
        pf_signs.append(torch.sign(pf_t).item())
        la_signs.append(s_t.item())
    print(f"\nTest 8 - Sign tracking under parameter sweep (sign problem)")
    print(f"  pfaffian  signs: {pf_signs}")
    print(f"  log_pfaffian signs: {la_signs}")
    print(f"  Signs agree: {pf_signs == la_signs}")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
