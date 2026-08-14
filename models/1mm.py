"""Single-matrix polynomial model."""

from __future__ import annotations

import os

import numpy as np
import torch

from matrix_hmc_track import config
from matrix_hmc_track.models.base import MatrixModel

model_name = "1mm"


def build_model(args):
    return OneMatrixPolynomialModel(
        ncol=args.ncol,
        couplings=args.coupling,
    )


class OneMatrixPolynomialModel(MatrixModel):
    """Single Hermitian matrix model with a polynomial potential.

    The action is:

    .. math::

        S = N \\sum_{n=1}^{P} t_n \\, \\mathrm{Tr}(X^n)

    where the coupling list ``[t_1, t_2, ..., t_P]`` determines the polynomial
    degree and shape of the potential.

    Args:
        ncol: Matrix size ``N``.
        couplings: List of polynomial coefficients ``[t_1, t_2, ..., t_P]``
            (must be non-empty).

    Raises:
        ValueError: If *couplings* is empty.
    """

    model_name = model_name

    def __init__(self, ncol: int, couplings: list) -> None:
        if len(couplings) == 0:
            raise ValueError("1MM model requires at least one coupling via --coupling t1 [t2 ...]")
        super().__init__(nmat=1, ncol=ncol)
        self.couplings = couplings
        self.is_hermitian = True
        self.is_traceless = False
        self._coupling_tensor = torch.tensor(couplings, dtype=config.real_dtype, device=config.device)

    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        X = self._resolve_X(X)
        mat = X[0]
        total = torch.zeros((), dtype=config.real_dtype, device=mat.device)
        for power, coeff in enumerate(self._coupling_tensor, start=1):
            if coeff == 0:
                continue
            trace_power = torch.trace(torch.linalg.matrix_power(mat, power)).real
            total = total + coeff.type_as(trace_power) * trace_power
        return self.ncol * total

    def measure_observables(self, X: torch.Tensor | None = None) -> tuple:
        with torch.no_grad():
            mat = self._resolve_X(X)[0]
            eigs = [torch.linalg.eigvalsh(mat).cpu().numpy()]
            trace_powers = []
            for power in range(1, len(self.couplings) + 1):
                trace_val = torch.trace(torch.linalg.matrix_power(mat, power)).cpu().numpy()
                trace_powers.append(trace_val)
            corrs = np.array(trace_powers, dtype=np.complex128)
        return eigs, corrs
    
    def load_fresh(self) -> None:
        """Load a fresh configuration (zero matrices)."""
        X = torch.stack([-2 * torch.eye(self.ncol, dtype=config.dtype, device=config.device) for _ in range(self.nmat)])
        # X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        self.set_state(X)

    def build_paths(
        self,
        name_prefix: str,
        data_path: str,
        *,
        step_size: float | None = None,
        nsteps: int | None = None,
        niters: int | None = None,
    ) -> dict[str, str]:
        coupling_suffix = "_".join(
            f"t{idx + 1}-{float(c):g}" for idx, c in enumerate(self.couplings)
        )
        hmc_suffix = self._hmc_path_suffix(step_size=step_size, nsteps=nsteps, niters=niters)
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{self._path_model_name()}_N{self.ncol}_{coupling_suffix}{hmc_suffix}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }

    def extra_config_lines(self) -> list[str]:
        couplings_str = ", ".join(f"t_{i+1}={c}" for i, c in enumerate(self.couplings))
        return [f"  Couplings t_n            = {couplings_str}"]

    def run_metadata(self) -> dict[str, object]:
        meta = super().run_metadata()
        meta.update(
            {
                "polynomial_degree": len(self.couplings),
                "model_variant": "1mm_polynomial",
            }
        )
        return meta
