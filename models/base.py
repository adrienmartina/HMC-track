"""Base class for matrix models used with the HMC driver."""

from __future__ import annotations

import os

import torch

from matrix_hmc_track import config
from matrix_hmc_track.algebra import get_eye_cached, makeH


class MatrixModel:
    """Abstract base class for matrix models used with the HMC driver.

    Subclasses must implement :meth:`potential` and :meth:`measure_observables`.
    All other methods provide sensible defaults that can be overridden as needed.

    Attributes:
        nmat: Number of matrices in the configuration (i.e. the dimension ``D``).
        ncol: Matrix size ``N`` (each matrix is ``N x N``).
        couplings: List of model coupling constants.  Set by the subclass.
        is_hermitian: Whether the matrices are constrained to be Hermitian.
            When ``True``, :meth:`force` projects the gradient to the Hermitian
            part automatically.
        is_traceless: Whether the matrices are constrained to be traceless.
            When ``True``, :meth:`force` removes the trace component of the
            gradient automatically.
    """

    def __init__(self, nmat: int, ncol: int) -> None:
        self.nmat = nmat
        self.ncol = ncol
        self.couplings = None
        self.is_hermitian = None
        self.is_traceless = None
        self._X: torch.Tensor | None = None

    def _resolve_X(self, X: torch.Tensor | None = None) -> torch.Tensor:
        if X is not None:
            return X
        if self._X is None:
            raise ValueError("Model configuration has not been initialized")
        return self._X

    def set_state(self, X: torch.Tensor) -> None:
        """Store *X* as the current field configuration.

        Args:
            X: Configuration tensor of shape ``(nmat, N, N)``.
        """
        self._X = X

    def get_state(self) -> torch.Tensor:
        """Return the current field configuration tensor.

        Returns:
            Tensor of shape ``(nmat, N, N)``.

        Raises:
            ValueError: If the configuration has not been set yet.
        """
        return self._resolve_X()

    def load_fresh(self) -> None:
        """Load a fresh (zero) configuration.

        The default initialises all matrices to zero.  Subclasses may override
        this to provide a physically motivated starting point (e.g. a fuzzy-sphere
        background or scaled random matrices).
        """
        X = torch.zeros((self.nmat, self.ncol, self.ncol), dtype=config.dtype, device=config.device)
        self.set_state(X)

    def initialize_configuration(self, ckpt_path: str, *, resume: bool = False) -> bool:
        """Load the field configuration from a checkpoint file or initialise fresh.

        Args:
            ckpt_path: Filesystem path to the checkpoint ``.pt`` file.
            resume: If ``True`` and *ckpt_path* exists, load that checkpoint;
                if the file is missing, fall through to a fresh initialisation.
                If ``False`` (default), always load fresh regardless of whether
                a checkpoint exists.

        Returns:
            ``True`` if a checkpoint was successfully loaded; ``False`` if a
            fresh configuration was initialised.
        """
        if resume:
            if os.path.isfile(ckpt_path):
                print("Reading old configuration file:", ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
                self.set_state(ckpt["X"].to(dtype=config.dtype, device=config.device))
                return True
            else:
                print("Configuration not found, loading fresh")
        else:
            print("Loading fresh configuration")

        self.load_fresh()

        return False

    def save_state(self, ckpt_path: str) -> None:
        """Persist the current configuration to a PyTorch checkpoint file.

        Args:
            ckpt_path: Destination file path (created or overwritten).
        """
        torch.save({"X": self.get_state()}, ckpt_path)

    def force(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the HMC force ``-dV/dX`` via autograd.

        Calls :meth:`potential` with ``requires_grad=True`` and uses
        :func:`torch.Tensor.backward` to obtain the gradient.  If
        :attr:`is_hermitian` is ``True`` the gradient is projected to its
        Hermitian part; if :attr:`is_traceless` is ``True`` the trace is
        removed, ensuring the force lies in the correct Lie-algebra subspace.

        Args:
            X: Configuration to differentiate at.  Defaults to the current
                internal state when ``None``.

        Returns:
            Force tensor of the same shape as ``X``.
        """
        X = self._resolve_X(X)
        Y = X.detach().requires_grad_(True)
        pot = self.potential(Y)
        pot.backward()
        res = Y.grad
        if self.is_hermitian:
            res = makeH(res)
        if self.is_traceless:
            trs = torch.diagonal(res, dim1=-2, dim2=-1).sum(-1).real / self.ncol
            eye = get_eye_cached(self.ncol, device=res.device, dtype=res.dtype)
            res = res - trs[..., None, None] * eye
        return res

    def status_string(self, X: torch.Tensor | None = None) -> str:
        """Return a one-line summary of the configuration for logging.

        The default implementation reports the per-matrix average of
        ``Tr(X_i^2)``.  Subclasses may override this to include additional
        physical observables.

        Args:
            X: Configuration to summarise.  Defaults to the current internal state.

        Returns:
            Human-readable string, e.g. ``"trX_i^2 = 1.23456. "``
        """
        X = self._resolve_X(X)
        avg_tr = (torch.einsum("bij,bji->", X, X).real / (self.nmat * self.ncol)).item()
        return f"trX_i^2 = {avg_tr:.5f}. "

    @staticmethod
    def _format_path_value(value: float) -> str:
        return str(round(float(value), 10))

    def _path_model_name(self) -> str:
        model_name = getattr(self, "model_name", self.__class__.__name__.lower())
        if getattr(self, "bosonic", False):
            return f"bos{model_name}"
        return model_name

    def _hmc_path_suffix(
        self,
        *,
        step_size: float | None = None,
        nsteps: int | None = None,
        niters: int | None = None,
    ) -> str:
        parts = []
        if step_size is not None:
            parts.append(f"dt{self._format_path_value(step_size)}")
        if nsteps is not None:
            parts.append(f"steps{int(nsteps)}")
        if niters is not None:
            parts.append(f"iters{int(niters)}")
        return f"_{'_'.join(parts)}" if parts else ""

    def build_paths(
        self,
        name_prefix: str,
        data_path: str,
        *,
        step_size: float | None = None,
        nsteps: int | None = None,
        niters: int | None = None,
    ) -> dict[str, str]:
        """Construct the file-system paths used for this run's outputs.

        Args:
            name_prefix: Short label prepended to the run directory name.
            data_path: Root directory under which all run subdirectories live.

        Returns:
            Dictionary with keys ``'dir'``, ``'eigs'``, ``'corrs'``,
            ``'meta'``, and ``'ckpt'`` mapping to absolute paths.
        """
        model_name = self._path_model_name()
        hmc_suffix = self._hmc_path_suffix(step_size=step_size, nsteps=nsteps, niters=niters)
        run_dir = os.path.join(
            data_path,
            f"{name_prefix}_{model_name}_D{self.nmat}_N{self.ncol}{hmc_suffix}",
        )
        return {
            "dir": run_dir,
            "eigs": os.path.join(run_dir, "evals.npz"),
            "corrs": os.path.join(run_dir, "corrs.npz"),
            "meta": os.path.join(run_dir, "metadata.json"),
            "ckpt": os.path.join(run_dir, "checkpoint.pt"),
        }
    
    def potential(self, X: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the action/potential ``V(X)`` for the given configuration.

        Must be differentiable with respect to ``X`` (used by :meth:`force`
        via autograd).

        Args:
            X: Configuration tensor of shape ``(nmat, N, N)``.  When ``None``
                the current internal state is used.

        Returns:
            Scalar tensor representing the potential energy.

        Raises:
            NotImplementedError: Must be overridden by every concrete subclass.
        """
        raise NotImplementedError

    def measure_observables(self, X: torch.Tensor | None = None) -> tuple:
        """Measure and return observables for the current configuration.

        Called once per HMC step by the simulation driver to collect statistics.

        Args:
            X: Configuration tensor of shape ``(nmat, N, N)``.  When ``None``
                the current internal state is used.

        Returns:
            Tuple ``(eigs, corrs)`` where *eigs* is a list of eigenvalue arrays
            (one per matrix or composite observable) and *corrs* is either a
            1-D NumPy array of scalar correlators or ``None``.

        Raises:
            NotImplementedError: Must be overridden by every concrete subclass.
        """
        raise NotImplementedError

    def extra_config_lines(self) -> list[str]:
        """Return optional human-readable configuration lines for the run header.

        Override to include model-specific parameters (couplings, flags, etc.)
        in the startup printout produced by :func:`~matrix_hmc_track.simulation.run`.

        Returns:
            List of strings, each printed as a single line.  Empty by default.
        """
        return []

    def run_metadata(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary of model metadata for logging.

        Written to ``metadata.json`` at the start of each run.  Subclasses
        should call ``super().run_metadata()`` and ``update`` the result with
        any additional parameters.

        Returns:
            Dictionary containing at least ``model_name``, ``nmat``, ``ncol``,
            ``couplings``, and ``dtype``.
        """
        return {
            "model_name": getattr(self, "model_name", self.__class__.__name__.lower()),
            "nmat": self.nmat,
            "ncol": self.ncol,
            "couplings": self.couplings,
            "dtype": str(config.dtype),
        }
