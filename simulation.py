"""HMC simulation runner — the main public entry point for library use."""

from __future__ import annotations

import cProfile
import datetime
import importlib
import importlib.util
import json
import math
import os
import pstats
import random
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from matrix_hmc_track import config as _config
from matrix_hmc_track.algebra import get_eye_cached, makeH, random_hermitian
from matrix_hmc_track.hmc import HMCParams, hamil, update
from matrix_hmc_track.models.base import MatrixModel

_BUILTIN_MODEL_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_module(model_name_or_path: str):
    """Return a model module by built-in name or file path.

    If the argument contains a path separator or ends in .py it is loaded
    directly from disk; otherwise it is looked up in the built-in models dir.
    """
    is_path = "/" in model_name_or_path or os.sep in model_name_or_path or model_name_or_path.endswith(".py")

    if is_path:
        p = Path(model_name_or_path)
        if not p.exists():
            raise ValueError(f"Model file not found: {p}")
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    if (_BUILTIN_MODEL_DIR / f"{model_name_or_path}.py").exists():
        return importlib.import_module(f"matrix_hmc_track.models.{model_name_or_path}")

    known = sorted(p.stem for p in _BUILTIN_MODEL_DIR.glob("*.py") if not p.stem.startswith("_"))
    raise ValueError(
        f"Unknown model '{model_name_or_path}'. Built-in models: {known}. "
        "To use a custom model, pass the path: --model ./my_model.py"
    )


# ---------------------------------------------------------------------------
# I/O helpers (intentionally private — details that callers shouldn't see)
# ---------------------------------------------------------------------------

def _ensure_output_slots(paths: Iterable[str], *, force: bool, allow_existing: bool) -> None:
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not (force or allow_existing):
        raise FileExistsError(
            "Output files already exist:\n" + "\n".join(existing) +
            "\nUse force=True to overwrite or resume=True to append."
        )
    if force:
        for path in existing:
            os.remove(path)


def _relocate_paths(paths: dict[str, str], run_dir: str) -> dict[str, str]:
    original_dir = paths["dir"]
    relocated = dict(paths)
    relocated["dir"] = run_dir

    original_abs = os.path.abspath(original_dir)
    for key, path in paths.items():
        if key == "dir":
            continue
        path_abs = os.path.abspath(path)
        try:
            is_inside_run_dir = os.path.commonpath([original_abs, path_abs]) == original_abs
        except ValueError:
            is_inside_run_dir = False
        if is_inside_run_dir:
            relocated[key] = os.path.join(run_dir, os.path.relpath(path_abs, original_abs))

    return relocated


def _prepare_run_directory(paths: dict[str, str], *, force: bool, resume: bool) -> dict[str, str]:
    """Create the output directory, choosing *_runN for fresh colliding runs."""
    if resume:
        os.makedirs(paths["dir"], exist_ok=True)
        return paths

    base_dir = paths["dir"]
    run_number = 1
    while True:
        run_dir = base_dir if run_number == 1 else f"{base_dir}_run{run_number}"
        try:
            os.makedirs(run_dir, exist_ok=False)
        except FileExistsError:
            run_number += 1
            continue
        return _relocate_paths(paths, run_dir)


def _prepare_matrix_snapshot_dir(run_dir: str, *, force: bool, allow_existing: bool) -> tuple[str, int]:
    target = os.path.join(run_dir, "all_mats")
    if os.path.exists(target):
        if force:
            shutil.rmtree(target)
        elif not allow_existing:
            raise FileExistsError(
                f"Matrix snapshot directory {target} already exists. "
                "Use force=True or resume=True."
            )
    os.makedirs(target, exist_ok=True)
    max_end = 0
    if allow_existing:
        for name in os.listdir(target):
            if not name.endswith(".npy") or "_" not in name[:-4]:
                continue
            start_str, end_str = name[:-4].split("_", 1)
            if start_str.isdigit() and end_str.isdigit():
                max_end = max(max_end, int(end_str))
    return target, max_end


def _append_npz(path: str, values: np.ndarray, *, key: str, dtype: np.dtype) -> None:
    if values.size == 0:
        return
    new_values = np.asarray(values, dtype=dtype)
    if os.path.exists(path):
        with np.load(path) as existing:
            new_values = np.concatenate((existing[key], new_values), axis=0)
    np.savez(path, **{key: new_values})


def _append_step_eigs_npz(
    path: str,
    values: np.ndarray,
    accepted: np.ndarray,
    *,
    labels: Iterable[str],
) -> None:
    if values.size == 0:
        return

    new_values = np.asarray(values, dtype=np.float64)
    new_accepted = np.asarray(accepted, dtype=np.bool_)
    if new_values.ndim != 4:
        raise ValueError(
            "Step eigenvalue data must have shape "
            f"(niters, nsteps, nchannels, ncol), got {new_values.shape}"
        )
    if new_values.shape[0] != new_accepted.shape[0]:
        raise ValueError(
            "Step eigenvalue acceptance flags must match the number of "
            f"trajectories, got {new_accepted.shape[0]} flags for {new_values.shape[0]} trajectories"
        )

    label_values = np.asarray(tuple(labels), dtype="U")
    if label_values.size == 0:
        label_values = np.asarray(
            [f"channel_{i}" for i in range(new_values.shape[2])],
            dtype="U",
        )
    if label_values.size != new_values.shape[2]:
        raise ValueError(
            f"Got {label_values.size} step eigenvalue labels for {new_values.shape[2]} channels"
        )

    if os.path.exists(path):
        with np.load(path) as existing:
            old_values = existing["values"]
            if old_values.shape[1:] != new_values.shape[1:]:
                raise ValueError(
                    f"Cannot append step eigenvalues with shape {new_values.shape[1:]} "
                    f"to existing shape {old_values.shape[1:]}"
                )
            new_values = np.concatenate((old_values, new_values), axis=0)
            if "accepted" in existing:
                old_accepted = existing["accepted"]
            else:
                old_accepted = np.full(old_values.shape[0], True, dtype=np.bool_)
            new_accepted = np.concatenate((old_accepted, new_accepted), axis=0)
            if "labels" in existing:
                label_values = existing["labels"].astype("U")

    np.savez(path, values=new_values, accepted=new_accepted, labels=label_values)


def _append_escape_npz(
    path: str,
    values: dict[str, np.ndarray | float | int | bool],
    *,
    reference_trx2: float,
    tolerance_atol: float,
    tolerance_rtol: float,
) -> None:
    escaped = np.asarray(values["escaped"], dtype=np.bool_)
    if escaped.size == 0:
        return

    payload = {
        "iteration": np.asarray(values["iteration"], dtype=np.int64),
        "escaped": escaped,
        "in_initial_basin": np.asarray(values["in_initial_basin"], dtype=np.bool_),
        "trx2_flow": np.asarray(values["trx2_flow"], dtype=np.float64),
        "grad_norm": np.asarray(values["grad_norm"], dtype=np.float64),
        "potential": np.asarray(values["potential"], dtype=np.float64),
        "converged": np.asarray(values["converged"], dtype=np.bool_),
        "descent_steps": np.asarray(values["descent_steps"], dtype=np.int64),
        "reference_trx2": np.asarray(reference_trx2, dtype=np.float64),
        "tolerance_atol": np.asarray(tolerance_atol, dtype=np.float64),
        "tolerance_rtol": np.asarray(tolerance_rtol, dtype=np.float64),
    }
    optional_dtypes = {
        "primary_escaped": np.bool_,
        "primary_trx2_flow": np.float64,
        "validation_passed": np.bool_,
        "validation_converged": np.bool_,
        "validation_halvings": np.int64,
    }
    for key, dtype in optional_dtypes.items():
        if key in values:
            payload[key] = np.asarray(values[key], dtype=dtype)

    if os.path.exists(path):
        with np.load(path) as existing:
            for key in (
                "iteration",
                "escaped",
                "in_initial_basin",
                "trx2_flow",
                "grad_norm",
                "potential",
                "converged",
                "descent_steps",
            ):
                payload[key] = np.concatenate((existing[key], payload[key]), axis=0)
            for key in optional_dtypes:
                if key in payload and key in existing:
                    payload[key] = np.concatenate((existing[key], payload[key]), axis=0)
            payload["reference_trx2"] = existing["reference_trx2"]
            payload["tolerance_atol"] = existing["tolerance_atol"]
            payload["tolerance_rtol"] = existing["tolerance_rtol"]

    np.savez(path, **payload)


def _flush_buffers(ev_buf: list, corr_buf: list, paths: dict[str, str]) -> None:
    if ev_buf:
        _append_npz(paths["eigs"], np.stack(ev_buf).astype(np.complex128),
                    key="values", dtype=np.complex128)
        ev_buf.clear()
    if corr_buf:
        _append_npz(paths["corrs"], np.stack(corr_buf).astype(np.complex128),
                    key="values", dtype=np.complex128)
        corr_buf.clear()


def _flush_step_eig_buffers(
    step_eig_buf: list[np.ndarray],
    step_accept_buf: list[bool],
    path: str | None,
    labels: Iterable[str],
) -> None:
    if not step_eig_buf or path is None:
        return
    _append_step_eigs_npz(
        path,
        np.stack(step_eig_buf, axis=0),
        np.asarray(step_accept_buf, dtype=np.bool_),
        labels=labels,
    )
    step_eig_buf.clear()
    step_accept_buf.clear()


def _flush_escape_buffers(
    escape_buf: dict[str, list],
    path: str | None,
    *,
    reference_trx2: float | None,
    tolerance_atol: float,
    tolerance_rtol: float,
) -> None:
    if path is None or reference_trx2 is None or not escape_buf["iteration"]:
        return
    _append_escape_npz(
        path,
        {key: np.asarray(value) for key, value in escape_buf.items()},
        reference_trx2=reference_trx2,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
    )
    for value in escape_buf.values():
        value.clear()


def _project_to_model_constraints(model: MatrixModel, X: torch.Tensor) -> torch.Tensor:
    if getattr(model, "is_hermitian", False):
        X = makeH(X)
    if getattr(model, "is_traceless", False):
        traces = torch.diagonal(X, dim1=-2, dim2=-1).sum(-1).real / model.ncol
        eye = get_eye_cached(model.ncol, device=X.device, dtype=X.dtype)
        X = X - traces[..., None, None] * eye
    return X


def _safe_potential_value(model: MatrixModel, X: torch.Tensor) -> float:
    with torch.enable_grad():
        value = model.potential(X)
    return float(value.detach().real.item())


def _tr_x123_sq_over_n(X: torch.Tensor) -> float:
    upto = min(3, X.shape[0])
    if upto == 0:
        return 0.0
    X123 = X[:upto]
    value = torch.einsum("bij,bji->", X123, X123).real / X.shape[-1]
    return float(value.detach().cpu().item())


def _classical_gradient_descent(
    model: MatrixModel,
    X: torch.Tensor,
    *,
    max_steps: int,
    step_size: float,
    grad_tol: float,
    min_step_size: float,
    max_backtracks: int,
    armijo: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Descend the classical potential from *X* without mutating model state."""
    Y = _project_to_model_constraints(model, X.detach().clone())
    potential = _safe_potential_value(model, Y)
    grad_norm = math.inf
    converged = False
    steps_taken = 0

    for step_index in range(1, max_steps + 1):
        grad = model.force(Y)
        grad = _project_to_model_constraints(model, grad)
        grad_sq = (grad.conj() * grad).real.sum()
        grad_norm = float(torch.sqrt(torch.clamp(grad_sq, min=0.0)).detach().cpu().item())
        if not math.isfinite(grad_norm):
            break
        if grad_norm <= grad_tol:
            converged = True
            steps_taken = step_index - 1
            break

        accepted = False
        local_step = step_size
        for _ in range(max_backtracks):
            proposal = _project_to_model_constraints(model, Y - local_step * grad)
            proposal_potential = _safe_potential_value(model, proposal)
            sufficient_decrease = potential - armijo * local_step * grad_norm * grad_norm
            if math.isfinite(proposal_potential) and proposal_potential <= sufficient_decrease:
                Y = proposal.detach()
                potential = proposal_potential
                accepted = True
                steps_taken = step_index
                break
            local_step *= 0.5
            if local_step < min_step_size:
                break

        if not accepted:
            steps_taken = step_index - 1
            break

    return Y, {
        "potential": potential,
        "grad_norm": grad_norm,
        "converged": converged,
        "steps": steps_taken,
        "trx2": _tr_x123_sq_over_n(Y),
    }


def _load_escape_reference(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    with np.load(path) as existing:
        if "reference_trx2" not in existing:
            return None
        return float(np.asarray(existing["reference_trx2"]).item())


def _make_escape_buffer() -> dict[str, list]:
    return {
        "iteration": [],
        "escaped": [],
        "in_initial_basin": [],
        "trx2_flow": [],
        "grad_norm": [],
        "potential": [],
        "converged": [],
        "descent_steps": [],
        "primary_escaped": [],
        "primary_trx2_flow": [],
        "validation_passed": [],
        "validation_converged": [],
        "validation_halvings": [],
    }


def _recorded_leapfrog(
    X: torch.Tensor,
    hmc_params: HMCParams,
    model: MatrixModel,
) -> tuple[torch.Tensor, float, float, list[torch.Tensor]]:
    """Run the original leapfrog arithmetic while copying step snapshots.

    This intentionally mirrors :func:`matrix_hmc_track.hmc.leapfrog`; the only extra
    work during integration is cloning the already-computed positions.  Eigenvalue
    calculations are done after Metropolis has finished.
    """
    dt_local = hmc_params.dt
    begin_traj = getattr(model, "begin_trajectory", None)
    if callable(begin_traj):
        begin_traj(X)

    mom_X = random_hermitian(
        model.ncol,
        traceless=bool(model.is_traceless),
        batchsize=model.nmat,
    )
    ham_init = hamil(X, mom_X, model)
    step_snapshots: list[torch.Tensor] = []

    X = X + 0.5 * dt_local * mom_X

    for _ in range(1, hmc_params.nsteps):
        f_X = model.force(X)
        mom_X = mom_X - dt_local * f_X
        X = X + dt_local * mom_X
        step_snapshots.append(X.detach().to(device="cpu", copy=True))

    f_X = model.force(X)
    mom_X = mom_X - dt_local * f_X
    X = X + 0.5 * dt_local * mom_X
    step_snapshots.append(X.detach().to(device="cpu", copy=True))

    ham_final = hamil(X, mom_X, model)
    return X, ham_init, ham_final, step_snapshots


def _update_with_step_eig_recording(
    acc_count: int,
    hmc_params: HMCParams,
    model: MatrixModel,
    measure_step_eigenvalues,
    reject_prob: float = 1.0,
    verbose: bool = True,
    emit=None,
) -> tuple[int, np.ndarray, bool]:
    """Run one HMC update and return per-step eigenvalues.

    The Metropolis evolution matches :func:`matrix_hmc_track.hmc.update`; measurements
    are evaluated from detached snapshots after the accept/reject decision.
    """
    X = model.get_state()
    X_bak = X.clone()
    X_new, H0, H1, step_snapshots = _recorded_leapfrog(X, hmc_params, model)
    dH = H1 - H0
    finite_h0 = np.isfinite(H0)
    finite_h1 = np.isfinite(H1)
    finite_dh = np.isfinite(dH)

    accept = bool(finite_h0 and finite_h1 and finite_dh)
    if accept and reject_prob > 0.0:
        r = random.uniform(0.0, reject_prob)
        if dH > 0.0:
            accept = (-dH) > math.log(r)

    sink = print if emit is None else emit

    if accept:
        model.set_state(X_new)
        acc_count += 1
        if verbose:
            sink(f"ACCEPT: dH={dH: 8.3f}, expDH={np.exp(-dH): 8.3f}, H0={H0: 8.4f},  {model.status_string()}")
    else:
        model.set_state(X_bak)
        if verbose:
            if finite_h0 and finite_h1 and finite_dh:
                sink(f"REJECT: dH={dH: 8.3f}, expDH={np.exp(-dH): 8.3f}, H0={H0: 8.4f},  {model.status_string()}")
            else:
                sink(
                    "REJECT: non-finite Hamiltonian encountered "
                    f"(H0={H0}, H1={H1}, dH={dH}),  {model.status_string()}"
                )

    end_traj = getattr(model, "end_trajectory", None)
    if callable(end_traj):
        end_traj(accept)

    step_values = np.stack(
        [np.asarray(measure_step_eigenvalues(snapshot), dtype=np.float64) for snapshot in step_snapshots],
        axis=0,
    )
    return acc_count, step_values, accept


def _create_snapshot_chunk(
    directory: str, *, start: int, count: int, dtype: np.dtype, shape_tail: tuple
) -> np.memmap:
    end = start + count - 1
    filename = os.path.join(directory, f"{start:08d}_{end:08d}.npy")
    return np.lib.format.open_memmap(filename, mode="w+", dtype=dtype, shape=(count, *shape_tail))


def _write_metadata(path: str, model: MatrixModel, **run_kwargs) -> None:
    summary = {
        "model": model.run_metadata(),
        "run": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                for k, v in run_kwargs.items()},
        "runtime": {
            "device": str(_config.device),
            "dtype": str(_config.dtype),
            "num_threads": _config.CPU_NUM_THREADS,
            "num_interop_threads": _config.CPU_NUM_INTEROP_THREADS,
            "timestamp": datetime.datetime.now().isoformat(),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _write_run_stats(path: str, stats: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def _jsonable_model_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return str(value.shape)
    if isinstance(value, (list, tuple)):
        return [_jsonable_model_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_model_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _base_run_stats(
    model: MatrixModel,
    *,
    paths: dict[str, str],
    niters: int,
    step_size: float,
    nsteps: int,
    output: str | Path,
    name: str,
    save_every: int,
    save_checkpoints: bool,
    save_matrices: bool,
    save_evals: bool,
    save_step_eigenvalues: bool,
    track_escape: bool,
    escape_descent_steps: int,
    escape_descent_step_size: float,
    escape_validation_halvings: int,
    escape_grad_tol: float,
    escape_trx2_atol: float,
    escape_trx2_rtol: float,
    escape_min_step_size: float,
    escape_max_backtracks: int,
    escape_persistence: int,
    stop_on_escape: bool,
    resume: bool,
    force: bool,
    seed: int | None,
    dry_run: bool,
    quiet_trajectories: bool,
) -> dict[str, object]:
    couplings = getattr(model, "couplings", None)
    return {
        "model": getattr(model, "model_name", model.__class__.__name__),
        "ncol": int(model.ncol),
        "nmat": int(model.nmat),
        "g": _jsonable_model_value(getattr(model, "g", None)),
        "omega": _jsonable_model_value(getattr(model, "omega", None)),
        "couplings": _jsonable_model_value(couplings),
        "bosonic": bool(getattr(model, "bosonic", False)),
        "spin": _jsonable_model_value(getattr(model, "spin", None)),
        "lorentzian": bool(getattr(model, "lorentzian", False)),
        "niters_requested": int(niters),
        "completed_trajectories": 0,
        "accepted_trajectories": 0,
        "acceptance_rate": 0.0,
        "step_size": float(step_size),
        "nsteps": int(nsteps),
        "dt": float(step_size / nsteps),
        "seed": seed,
        "output": str(output),
        "name": str(name),
        "run_dir": paths["dir"],
        "save_every": int(save_every),
        "save_checkpoints": bool(save_checkpoints),
        "save_matrices": bool(save_matrices),
        "save_evals": bool(save_evals),
        "save_step_eigenvalues": bool(save_step_eigenvalues),
        "step_eigenvalues_disabled": not bool(save_step_eigenvalues),
        "track_escape": bool(track_escape),
        "stop_on_escape": bool(stop_on_escape),
        "stop_on_escape_fired": False,
        "escape_persistence": int(escape_persistence),
        "resume": bool(resume),
        "force": bool(force),
        "dry_run": bool(dry_run),
        "quiet_trajectories": bool(quiet_trajectories),
        "run_complete": False,
        "escape_descent_settings": {
            "max_steps": int(escape_descent_steps),
            "step_size": float(escape_descent_step_size),
            "validation_halvings": int(escape_validation_halvings),
            "grad_tol": float(escape_grad_tol),
            "trx2_atol": float(escape_trx2_atol),
            "trx2_rtol": float(escape_trx2_rtol),
            "min_step_size": float(escape_min_step_size),
            "max_backtracks": int(escape_max_backtracks),
        },
        "escape_reference": None,
        "escape_reference_converged": None,
        "escape_first_iteration": None,
        "escape_first_raw_iteration": None,
        "escape_first_reliable_iteration": None,
        "escape_first_persistent_iteration": None,
        "escape_persistent_detected": False,
        "escape_detected": False,
        "escape_reliable_detected": False,
        "escape_censored": bool(track_escape),
        "tau_int": None,
        "tau_int_window": None,
        "tau_int_samples": None,
        "tau_int_warnings": None,
        "tau_int_rel_error": None,
        "tau_esc": None,
        "tau_esc_censored": None,
        "escape_convergence_fraction": None,
        "escape_classification_reliable": None,
        "timestamp_start": datetime.datetime.now().isoformat(),
        "timestamp_end": None,
        "runtime_seconds": None,
    }


def _maybe_profile(enabled: bool) -> cProfile.Profile | None:
    if not enabled:
        return None
    p = cProfile.Profile()
    p.enable()
    return p


def _stop_profile(profiler: cProfile.Profile | None) -> None:
    if profiler is None:
        return
    profiler.disable()
    ps = pstats.Stats(profiler)
    ps.strip_dirs().sort_stats(pstats.SortKey.TIME)
    ps.print_stats(10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format a wall-clock duration with adaptive units."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m {rem:.2f}s"


def _r2_all_from_eigs(stacked: np.ndarray) -> float:
    """``R2_all = sum_{I=1}^{4} Tr X_I^2`` from one trajectory's stacked eigenvalues.

    Channels ``0..3`` hold the Hermitian spectra of ``X_1..X_4``; this matches
    ``load_R2_all`` in the escape-time analysis.
    """
    if stacked.ndim < 2 or stacked.shape[0] < 4:
        return float("nan")
    return float(np.sum(np.asarray(stacked[:4]).real ** 2))


def _autocorrelation_fft(x) -> np.ndarray:
    """Normalized autocorrelation function, normalized by pair counts ``n - k``."""
    values = np.asarray(x, dtype=np.float64)
    n = values.size
    if n == 0:
        return np.asarray([], dtype=np.float64)
    centered = values - np.mean(values)
    if float(np.dot(centered, centered)) == 0.0:
        acf = np.zeros(n, dtype=np.float64)
        acf[0] = 1.0
        return acf
    fft_len = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_len)
    acov = np.fft.irfft(transformed * np.conjugate(transformed), n=fft_len)[:n]
    acov /= np.arange(n, 0, -1, dtype=np.float64)
    return acov / acov[0]


def _integrated_autocorrelation_time(x, c: float = 5.0) -> dict:
    """Sokal-windowed integrated autocorrelation time in units of trajectories.

    ``tau_int = 1/2 + sum_{k=1}^{W} rho(k)``, with ``W`` the first lag satisfying
    ``W >= c * tau_int(W)``, capped at the last strictly positive lag.  Uncorrelated
    data gives ``tau_int = 1/2``; one independent configuration costs
    ``2 * tau_int`` trajectories.
    """
    acf = _autocorrelation_fft(x)
    n = int(acf.size)
    warnings: list[str] = []
    if n == 0:
        return {"n": 0, "window": 0, "tau_int": float("nan"), "warnings": ["empty series"]}
    if n < 50:
        warnings.append("len < 50")
    nonpositive = np.flatnonzero(acf[1:] <= 0.0)
    first_nonpositive = int(nonpositive[0] + 1) if nonpositive.size else n
    max_positive_window = max(0, first_nonpositive - 1)
    cumulative = 0.5 + np.cumsum(acf[1:])
    window = max_positive_window
    for lag in range(1, max_positive_window + 1):
        if lag >= c * float(cumulative[lag - 1]):
            window = lag
            break
    tau_int = max(0.5, 0.5 + float(np.sum(acf[1 : window + 1])))
    if n < 10 * tau_int:
        warnings.append("len < 10*tau_int")
    return {"n": n, "window": int(window), "tau_int": tau_int, "warnings": warnings}


def run(
    model: MatrixModel,
    *,
    niters: int = 100,
    step_size: float = 0.5,
    nsteps: int = 50,
    output: str | Path = "data",
    name: str = "run",
    save_every: int = 10,
    save_checkpoints: bool = True,
    save_matrices: bool = False,
    save_evals: bool = True,
    save_step_eigenvalues: bool = True,
    track_escape: bool = False,
    escape_descent_steps: int = 1000,
    escape_descent_step_size: float = 0.01,
    escape_validation_halvings: int = 0,
    escape_grad_tol: float = 1e-8,
    escape_trx2_atol: float = 1e-6,
    escape_trx2_rtol: float = 1e-5,
    escape_min_step_size: float = 1e-12,
    escape_max_backtracks: int = 25,
    escape_persistence: int = 1,
    stop_on_escape: bool = False,
    resume: bool = False,
    force: bool = False,
    seed: int | None = None,
    profile: bool = False,
    dry_run: bool = False,
    quiet_trajectories: bool = False,
) -> MatrixModel:
    """Run HMC trajectories for *model* and write observables to *output/name_.../*.

    Parameters
    ----------
    model:            A MatrixModel instance (already constructed with all params).
    niters:           Number of HMC trajectories.
    step_size:        Total leapfrog trajectory length.
    nsteps:           Leapfrog steps per trajectory (dt = step_size / nsteps).
    output:           Root directory for output files.
    name:             Prefix for the run subdirectory.
    save_every:       Flush observables (and checkpoint if save_checkpoints) every K steps.
    save_checkpoints: Write a checkpoint .pt file every save_every steps.
    save_matrices:    Also write raw matrix snapshots.
    save_evals:       Write per-trajectory eigenvalues to evals.npz.  Disabling it
                      changes no measurement: observables are still computed and
                      tau_int is estimated from the in-memory series.  It only skips
                      the file, so the pooled multi-replica tau_int of the escape-time
                      analysis (which reads evals.npz) becomes unreconstructible.
    save_step_eigenvalues:
                       Write per-leapfrog-step eigenvalue diagnostics when the model supports them.
    track_escape:     Run classical gradient descent after each HMC trajectory
                       and store whether the descended point stayed in the
                       initial basin.
    escape_descent_steps:
                       Maximum gradient descent steps for each basin test.
    escape_descent_step_size:
                       Initial descent step before backtracking.
    escape_validation_halvings:
                       For escaped classifications, require repeat descents
                       with step_size / 2**k, k=1..N, to also escape.
    escape_grad_tol:   Stop descent when the Frobenius norm of the gradient is below this.
    escape_trx2_atol:  Absolute tolerance for matching the initial descended TrX2.
    escape_trx2_rtol:  Relative tolerance for matching the initial descended TrX2.
    escape_persistence: Number of CONSECUTIVE escaped+converged classifications
                       required before an escape is declared.  The recorded escape
                       time is the FIRST trajectory of that run, not the last.  Any
                       other outcome (in-basin, or non-converged) resets the count.
                       1 (default) reproduces the previous single-event behaviour.
    stop_on_escape:    End the run immediately after an escape is declared.
    resume:           Append to existing output files and load checkpoint if present.
    force:            Overwrite existing output files without error.
    seed:             RNG seed for reproducibility.
    profile:          Enable cProfile and print top-10 hotspots at the end.
    dry_run:          Print configuration and return without running.
    quiet_trajectories:
                       Suppress per-trajectory accept/reject and periodic
                       progress lines. Summary and warning lines are retained.
    """
    hmc_params = HMCParams(dt=step_size / nsteps, nsteps=nsteps)

    paths = model.build_paths(
        name,
        str(output),
        step_size=step_size,
        nsteps=nsteps,
        niters=niters,
    )
    paths = _prepare_run_directory(paths, force=force, resume=resume)
    stats_path = os.path.join(paths["dir"], "run_stats.json")
    measure_step_eigenvalues = getattr(model, "measure_step_eigenvalues", None)
    if not save_step_eigenvalues or not callable(measure_step_eigenvalues):
        measure_step_eigenvalues = None
    step_eigs_path = paths.get("step_eigs") if measure_step_eigenvalues is not None else None
    step_eig_labels = tuple(getattr(model, "step_eigenvalue_labels", ()))
    escape_path = os.path.join(paths["dir"], "escape.npz") if track_escape else None
    output_slots = [paths["corrs"]]
    if save_evals:
        output_slots.insert(0, paths["eigs"])
    if step_eigs_path is not None:
        output_slots.append(step_eigs_path)
    if escape_path is not None:
        output_slots.append(escape_path)

    _ensure_output_slots(output_slots, force=force, allow_existing=resume)
    _write_metadata(paths["meta"], model, niters=niters, step_size=step_size,
                    nsteps=nsteps, output=str(output), name=name,
                    track_escape=track_escape,
                    escape_descent_steps=escape_descent_steps,
                    escape_descent_step_size=escape_descent_step_size,
                    escape_validation_halvings=escape_validation_halvings,
                    escape_grad_tol=escape_grad_tol,
                    escape_trx2_atol=escape_trx2_atol,
                    escape_trx2_rtol=escape_trx2_rtol,
                    save_evals=save_evals,
                    save_step_eigenvalues=save_step_eigenvalues,
                    escape_persistence=escape_persistence,
                    stop_on_escape=stop_on_escape)
    run_started_at = time.time()
    run_stats = _base_run_stats(
        model,
        paths=paths,
        niters=niters,
        step_size=step_size,
        nsteps=nsteps,
        output=output,
        name=name,
        save_every=save_every,
        save_checkpoints=save_checkpoints,
        save_matrices=save_matrices,
        save_evals=save_evals,
        save_step_eigenvalues=save_step_eigenvalues,
        track_escape=track_escape,
        escape_descent_steps=escape_descent_steps,
        escape_descent_step_size=escape_descent_step_size,
        escape_validation_halvings=escape_validation_halvings,
        escape_grad_tol=escape_grad_tol,
        escape_trx2_atol=escape_trx2_atol,
        escape_trx2_rtol=escape_trx2_rtol,
        escape_min_step_size=escape_min_step_size,
        escape_max_backtracks=escape_max_backtracks,
        escape_persistence=escape_persistence,
        stop_on_escape=stop_on_escape,
        resume=resume,
        force=force,
        seed=seed,
        dry_run=dry_run,
        quiet_trajectories=quiet_trajectories,
    )

    print("\n" + "=" * 52)
    print("  Matrix Model HMC — run configuration")
    print("=" * 52)
    print(f"  Model          {model.model_name}")
    print(f"  Matrix size N  {model.ncol}")
    for line in model.extra_config_lines():
        print(line)
    print(f"  Step size      {step_size}  ({nsteps} steps, dt = {step_size/nsteps:.4g})")
    print(f"  Device         {_config.device}  [{_config.dtype}]")
    print(f"  CPU threads    {_config.CPU_NUM_THREADS}/{_config.CPU_NUM_INTEROP_THREADS} (intra/inter-op)")
    print(f"  Checkpoint     {'every ' + str(save_every) + ' steps' if save_checkpoints else 'disabled'}")
    print(f"  Output dir     {paths['dir']}")
    if not save_evals:
        print("  Eigenvalues    not saved (--no-save-evals)")
    if step_eigs_path is not None:
        print(f"  Step evals     {step_eigs_path}")
    if escape_path is not None:
        print(f"  Escape data    {escape_path}")
    print("=" * 52 + "\n")

    if dry_run:
        run_stats["timestamp_end"] = datetime.datetime.now().isoformat()
        run_stats["runtime_seconds"] = time.time() - run_started_at
        _write_run_stats(stats_path, run_stats)
        return model

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    profiler = _maybe_profile(profile)
    resumed = model.initialize_configuration(paths["ckpt"], resume=resume)

    if not resumed:
        _ensure_output_slots(output_slots, force=True, allow_existing=False)

    reference_trx2: float | None = None
    if track_escape:
        reference_trx2 = _load_escape_reference(escape_path) if resumed and escape_path is not None else None
        if reference_trx2 is None:
            _, reference_info = _classical_gradient_descent(
                model,
                model.get_state(),
                max_steps=escape_descent_steps,
                step_size=escape_descent_step_size,
                grad_tol=escape_grad_tol,
                min_step_size=escape_min_step_size,
                max_backtracks=escape_max_backtracks,
            )
            reference_trx2 = float(reference_info["trx2"])
            run_stats["escape_reference_converged"] = bool(reference_info["converged"])
            print(
                "Escape reference: "
                f"TrX2_0={reference_trx2:.12g}, "
                f"grad_norm={float(reference_info['grad_norm']):.3g}, "
                f"steps={int(reference_info['steps'])}, "
                f"converged={bool(reference_info['converged'])}"
            )
            if not bool(reference_info["converged"]):
                print("WARNING: Escape reference descent did not converge; classifications are unreliable.")
        else:
            print(f"Escape reference loaded from existing escape data: TrX2_0={reference_trx2:.12g}")
            run_stats["escape_reference_converged"] = None
        run_stats["escape_reference"] = reference_trx2

    acc_count = 0
    ev_buf: list[np.ndarray] = []
    corr_buf: list[np.ndarray] = []
    step_eig_buf: list[np.ndarray] = []
    step_accept_buf: list[bool] = []
    escape_buf = _make_escape_buffer()
    snapshot_dir = None
    snapshot_offset = 0
    chunk: np.memmap | None = None
    escaped_once = False
    escape_streak = 0
    escape_streak_start: int | None = None
    completed_iters = 0
    escape_classifications = 0
    escape_converged_count = 0

    # R2_all series and its in-basin mask, used for the tau_int estimate below.
    r2_series: list[float] = []
    basin_flags: list[bool] = []

    # Hold the trajectory line back so the escape verdict and the elapsed time can
    # be appended to it before it is printed.
    pending_traj_line: list[str] = []
    traj_line_sink = pending_traj_line.append if not quiet_trajectories else None

    if save_matrices:
        snapshot_dir, snapshot_offset = _prepare_matrix_snapshot_dir(
            paths["dir"], force=force, allow_existing=resume
        )
        state_shape = model.get_state().shape
        state_dtype = np.dtype(model.get_state().detach().cpu().numpy().dtype)

    for i in range(1, niters + 1):
        completed_iters = i
        escape_verdict: str | None = None
        escape_declaration_msg: str | None = None
        descent_seconds: float | None = None
        in_basin_flag = True
        pending_traj_line.clear()
        traj_started_at = time.perf_counter()
        if measure_step_eigenvalues is None:
            acc_count = update(
                acc_count,
                hmc_params,
                model,
                verbose=not quiet_trajectories,
                emit=traj_line_sink,
            )
        else:
            acc_count, step_values, accepted = _update_with_step_eig_recording(
                acc_count,
                hmc_params,
                model,
                measure_step_eigenvalues,
                verbose=not quiet_trajectories,
                emit=traj_line_sink,
            )
            if step_values.shape[0] != nsteps:
                raise RuntimeError(
                    f"Expected {nsteps} step eigenvalue snapshots in trajectory {i}, "
                    f"got {step_values.shape[0]}"
                )
            step_eig_buf.append(step_values)
            step_accept_buf.append(accepted)

        traj_seconds = time.perf_counter() - traj_started_at

        if snapshot_dir is not None:
            global_i = snapshot_offset + i
            if chunk is None or (global_i - 1) % save_every == 0:
                if chunk is not None:
                    chunk.flush()
                remaining = niters - i + 1
                chunk = _create_snapshot_chunk(
                    snapshot_dir, start=global_i,
                    count=min(save_every, remaining),
                    dtype=state_dtype, shape_tail=state_shape,
                )
            state = model.get_state().detach()
            if state.device.type != "cpu":
                state = state.to("cpu")
            chunk[global_i - (snapshot_offset + (i - 1) // save_every * save_every) - 1] = state.numpy()

        eigs, corrs = model.measure_observables()
        stacked_eigs = np.stack(eigs)
        if save_evals:
            ev_buf.append(stacked_eigs)
        r2_series.append(_r2_all_from_eigs(stacked_eigs))
        if corrs is not None:
            corr_buf.append(corrs)

        if track_escape and escape_path is not None and reference_trx2 is not None:
            descent_started_at = time.perf_counter()
            _, escape_info = _classical_gradient_descent(
                model,
                model.get_state(),
                max_steps=escape_descent_steps,
                step_size=escape_descent_step_size,
                grad_tol=escape_grad_tol,
                min_step_size=escape_min_step_size,
                max_backtracks=escape_max_backtracks,
            )
            flow_trx2 = float(escape_info["trx2"])
            tolerance = escape_trx2_atol + escape_trx2_rtol * abs(reference_trx2)
            in_initial_basin = bool(abs(flow_trx2 - reference_trx2) <= tolerance)
            primary_escaped = not in_initial_basin
            escaped = primary_escaped
            converged = bool(escape_info["converged"])
            validation_passed = True
            validation_converged = True

            if primary_escaped and converged and escape_validation_halvings:
                validation_flow_trx2 = flow_trx2
                for halving in range(1, escape_validation_halvings + 1):
                    _, validation_info = _classical_gradient_descent(
                        model,
                        model.get_state(),
                        max_steps=escape_descent_steps,
                        step_size=escape_descent_step_size / (2 ** halving),
                        grad_tol=escape_grad_tol,
                        min_step_size=escape_min_step_size,
                        max_backtracks=escape_max_backtracks,
                    )
                    validation_flow_trx2 = float(validation_info["trx2"])
                    validation_in_basin = bool(
                        abs(validation_flow_trx2 - reference_trx2) <= tolerance
                    )
                    validation_converged = validation_converged and bool(
                        validation_info["converged"]
                    )
                    validation_passed = (
                        validation_passed
                        and bool(validation_info["converged"])
                        and not validation_in_basin
                    )
                if not validation_passed:
                    flow_trx2 = validation_flow_trx2
                    in_initial_basin = True
                    escaped = False
                    converged = converged and validation_converged
            descent_seconds = time.perf_counter() - descent_started_at

            escape_classifications += 1
            in_basin_flag = bool(in_initial_basin and converged)
            escape_verdict = (
                "Undecided" if not converged else ("True" if escaped else "False")
            )
            if converged:
                escape_converged_count += 1
            if escaped and not converged:
                print(
                    "WARNING: Escape-like classification did not converge "
                    f"at iteration {i}; keeping the run marked unreliable."
                )
            # Persistence: declare an escape only after `escape_persistence`
            # consecutive escaped+converged classifications, and record the FIRST
            # trajectory of that run.  At the separatrix the committor is ~1/2, so
            # roughly half of all crossings recross; requiring a run rejects those.
            if escaped and converged:
                if escape_streak == 0:
                    escape_streak_start = int(i)
                escape_streak += 1
            else:
                escape_streak = 0
                escape_streak_start = None

            if (
                escape_streak >= escape_persistence
                and run_stats["escape_first_persistent_iteration"] is None
            ):
                run_stats["escape_first_persistent_iteration"] = int(escape_streak_start)
                run_stats["escape_persistent_detected"] = True
                if escape_persistence > 1:
                    run_stats["escape_first_iteration"] = int(escape_streak_start)
                    escape_declaration_msg = (
                        f"Escape declared at iteration {escape_streak_start}: "
                        f"{escape_persistence} consecutive escaped classifications "
                        f"({escape_streak_start}..{i})."
                    )
            escaped_once = bool(run_stats["escape_persistent_detected"])

            if escape_persistence > 1 and escaped and converged:
                escape_verdict = (
                    f"True ({min(escape_streak, escape_persistence)}/{escape_persistence})"
                )
            escape_buf["iteration"].append(i)
            escape_buf["escaped"].append(escaped)
            escape_buf["in_initial_basin"].append(in_initial_basin)
            escape_buf["trx2_flow"].append(flow_trx2)
            escape_buf["grad_norm"].append(float(escape_info["grad_norm"]))
            escape_buf["potential"].append(float(escape_info["potential"]))
            escape_buf["converged"].append(converged)
            escape_buf["descent_steps"].append(int(escape_info["steps"]))
            escape_buf["primary_escaped"].append(primary_escaped)
            escape_buf["primary_trx2_flow"].append(float(escape_info["trx2"]))
            escape_buf["validation_passed"].append(validation_passed)
            escape_buf["validation_converged"].append(validation_converged)
            escape_buf["validation_halvings"].append(int(escape_validation_halvings))
            if escaped and run_stats["escape_first_raw_iteration"] is None:
                run_stats["escape_first_raw_iteration"] = int(i)
                run_stats["escape_first_iteration"] = int(i)
                run_stats["escape_detected"] = True
                run_stats["escape_censored"] = False
            if escaped and converged and run_stats["escape_first_reliable_iteration"] is None:
                run_stats["escape_first_reliable_iteration"] = int(i)
                run_stats["escape_reliable_detected"] = True

        basin_flags.append(in_basin_flag)

        if pending_traj_line:
            line = pending_traj_line.pop()
            if escape_verdict is not None:
                line = f"{line.rstrip()} has escaped = {escape_verdict}"
            timing = f"HMC {_format_duration(traj_seconds)}"
            if descent_seconds is not None:
                timing += f" | descent {_format_duration(descent_seconds)}"
            print(f"{line.rstrip()}  [{timing}]")
        if escape_declaration_msg is not None:
            print(escape_declaration_msg)

        if i % save_every == 0:
            _flush_buffers(ev_buf, corr_buf, paths)
            _flush_step_eig_buffers(step_eig_buf, step_accept_buf, step_eigs_path, step_eig_labels)
            _flush_escape_buffers(
                escape_buf,
                escape_path,
                reference_trx2=reference_trx2,
                tolerance_atol=escape_trx2_atol,
                tolerance_rtol=escape_trx2_rtol,
            )
            if chunk is not None:
                chunk.flush()
            if not quiet_trajectories:
                print(f"Iteration {i}, acceptance = {acc_count/i:.3f}, " + model.status_string())
            run_stats["completed_trajectories"] = int(completed_iters)
            run_stats["accepted_trajectories"] = int(acc_count)
            run_stats["acceptance_rate"] = float(acc_count / max(completed_iters, 1))
            if escape_classifications:
                convergence_fraction = escape_converged_count / escape_classifications
                run_stats["escape_convergence_fraction"] = float(convergence_fraction)
                run_stats["escape_classification_reliable"] = bool(convergence_fraction == 1.0)
            _write_run_stats(stats_path, run_stats)
            if save_checkpoints:
                model.save_state(paths["ckpt"])

        if stop_on_escape and escaped_once:
            declared = run_stats["escape_first_persistent_iteration"]
            print(
                f"Escape declared at iteration {declared}; stopping early at "
                f"iteration {i} because stop_on_escape=True."
            )
            run_stats["stop_on_escape_fired"] = True
            break

    _flush_buffers(ev_buf, corr_buf, paths)
    _flush_step_eig_buffers(step_eig_buf, step_accept_buf, step_eigs_path, step_eig_labels)
    _flush_escape_buffers(
        escape_buf,
        escape_path,
        reference_trx2=reference_trx2,
        tolerance_atol=escape_trx2_atol,
        tolerance_rtol=escape_trx2_rtol,
    )
    if chunk is not None:
        chunk.flush()

    if acc_count / max(completed_iters, 1) < 0.5:
        print("WARNING: Acceptance rate is below 50%")

    run_stats["completed_trajectories"] = int(completed_iters)
    run_stats["accepted_trajectories"] = int(acc_count)
    run_stats["acceptance_rate"] = float(acc_count / max(completed_iters, 1))
    if track_escape:
        if escape_classifications:
            convergence_fraction = escape_converged_count / escape_classifications
            run_stats["escape_convergence_fraction"] = float(convergence_fraction)
            run_stats["escape_classification_reliable"] = bool(
                convergence_fraction == 1.0
                and run_stats["escape_reference_converged"] is not False
            )
        else:
            run_stats["escape_convergence_fraction"] = 0.0
            run_stats["escape_classification_reliable"] = False
        if escape_persistence > 1:
            run_stats["escape_censored"] = bool(not run_stats["escape_persistent_detected"])
        else:
            run_stats["escape_censored"] = bool(not run_stats["escape_detected"])
    # ------------------------------------------------------------------ #
    # tau_int : integrated autocorrelation time of R2_all = sum_I Tr X_I^2,
    #           measured on the in-basin segment only (the slowest global mode
    #           IS the escape, so including it would be circular).
    # tau_esc : first-passage time in trajectories; right-censored when the run
    #           finished without a reliable escape.
    # ------------------------------------------------------------------ #
    segment = [
        value
        for value, in_basin in zip(r2_series, basin_flags)
        if in_basin and math.isfinite(value)
    ]
    tau_info = _integrated_autocorrelation_time(segment)
    tau_int = float(tau_info["tau_int"])
    has_tau_int = bool(tau_info["n"]) and math.isfinite(tau_int)

    run_stats["tau_int"] = tau_int if has_tau_int else None
    run_stats["tau_int_window"] = int(tau_info["window"]) if has_tau_int else None
    run_stats["tau_int_samples"] = int(tau_info["n"])

    # sd[tau_int(W)] / tau_int ~ sqrt(2(2W+1)/N_tot).  A single run is one replica;
    # the pooled multi-replica estimate is the one to quote in an analysis.
    tau_int_rel_error = None
    if has_tau_int and tau_info["n"] > 0:
        tau_int_rel_error = math.sqrt(
            2.0 * (2.0 * int(tau_info["window"]) + 1.0) / float(tau_info["n"])
        )
        run_stats["tau_int_rel_error"] = float(tau_int_rel_error)
        if tau_int_rel_error > 0.15:
            tau_info["warnings"].append(f"single-run estimate, rel. error ~{tau_int_rel_error:.0%}")
    run_stats["tau_int_warnings"] = list(tau_info["warnings"])

    tau_esc_event = bool(track_escape and run_stats["escape_persistent_detected"])
    run_stats["tau_esc"] = (
        int(run_stats["escape_first_persistent_iteration"]) if tau_esc_event else None
    )
    run_stats["tau_esc_censored"] = bool(track_escape and not tau_esc_event)

    run_stats["timestamp_end"] = datetime.datetime.now().isoformat()
    run_stats["runtime_seconds"] = time.time() - run_started_at
    run_stats["run_complete"] = True
    _write_run_stats(stats_path, run_stats)

    _stop_profile(profiler)

    if has_tau_int:
        tau_int_text = (
            f"{tau_int:.4f} +/- {tau_int * tau_int_rel_error:.4f}  "
            f"(Sokal window {run_stats['tau_int_window']}, "
            f"{run_stats['tau_int_samples']} in-basin samples)"
        )
    else:
        tau_int_text = "n/a (no samples)"
    if not track_escape:
        tau_esc_text = "n/a (escape tracking off)"
    elif tau_esc_event:
        tau_esc_text = f"{run_stats['tau_esc']} trajectories"
        if escape_persistence > 1:
            tau_esc_text += f"  (first of {escape_persistence} consecutive)"
    else:
        tau_esc_text = f"> {completed_iters} trajectories (censored, no escape)"

    print("=" * 52)
    print(f"  tau_int  =  {tau_int_text}")
    print(f"  tau_esc  =  {tau_esc_text}")
    if has_tau_int and tau_esc_event and tau_int > 0.0:
        print(f"  tau_esc / tau_int  =  {run_stats['tau_esc'] / tau_int:.1f}  (dimensionless)")
    if tau_info["warnings"]:
        print(f"  WARNING (tau_int): {', '.join(tau_info['warnings'])}")
    print("=" * 52)

    return model


__all__ = ["run", "_load_model_module"]
