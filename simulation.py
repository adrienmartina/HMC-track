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
from matrix_hmc_track.algebra import random_hermitian
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

    if accept:
        model.set_state(X_new)
        acc_count += 1
        print(f"ACCEPT: dH={dH: 8.3f}, expDH={np.exp(-dH): 8.3f}, H0={H0: 8.4f}, ", model.status_string())
    else:
        model.set_state(X_bak)
        if finite_h0 and finite_h1 and finite_dh:
            print(f"REJECT: dH={dH: 8.3f}, expDH={np.exp(-dH): 8.3f}, H0={H0: 8.4f}, ", model.status_string())
        else:
            print(
                "REJECT: non-finite Hamiltonian encountered "
                f"(H0={H0}, H1={H1}, dH={dH}), ",
                model.status_string(),
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
    resume: bool = False,
    force: bool = False,
    seed: int | None = None,
    profile: bool = False,
    dry_run: bool = False,
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
    resume:           Append to existing output files and load checkpoint if present.
    force:            Overwrite existing output files without error.
    seed:             RNG seed for reproducibility.
    profile:          Enable cProfile and print top-10 hotspots at the end.
    dry_run:          Print configuration and return without running.
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
    measure_step_eigenvalues = getattr(model, "measure_step_eigenvalues", None)
    if not callable(measure_step_eigenvalues):
        measure_step_eigenvalues = None
    step_eigs_path = paths.get("step_eigs") if measure_step_eigenvalues is not None else None
    step_eig_labels = tuple(getattr(model, "step_eigenvalue_labels", ()))
    output_slots = [paths["eigs"], paths["corrs"]]
    if step_eigs_path is not None:
        output_slots.append(step_eigs_path)

    _ensure_output_slots(output_slots, force=force, allow_existing=resume)
    _write_metadata(paths["meta"], model, niters=niters, step_size=step_size,
                    nsteps=nsteps, output=str(output), name=name)

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
    if step_eigs_path is not None:
        print(f"  Step evals     {step_eigs_path}")
    print("=" * 52 + "\n")

    if dry_run:
        return model

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    profiler = _maybe_profile(profile)
    resumed = model.initialize_configuration(paths["ckpt"], resume=resume)

    if not resumed:
        _ensure_output_slots(output_slots, force=True, allow_existing=False)

    acc_count = 0
    ev_buf: list[np.ndarray] = []
    corr_buf: list[np.ndarray] = []
    step_eig_buf: list[np.ndarray] = []
    step_accept_buf: list[bool] = []
    snapshot_dir = None
    snapshot_offset = 0
    chunk: np.memmap | None = None

    if save_matrices:
        snapshot_dir, snapshot_offset = _prepare_matrix_snapshot_dir(
            paths["dir"], force=force, allow_existing=resume
        )
        state_shape = model.get_state().shape
        state_dtype = np.dtype(model.get_state().detach().cpu().numpy().dtype)

    for i in range(1, niters + 1):
        if measure_step_eigenvalues is None:
            acc_count = update(acc_count, hmc_params, model)
        else:
            acc_count, step_values, accepted = _update_with_step_eig_recording(
                acc_count,
                hmc_params,
                model,
                measure_step_eigenvalues,
            )
            if step_values.shape[0] != nsteps:
                raise RuntimeError(
                    f"Expected {nsteps} step eigenvalue snapshots in trajectory {i}, "
                    f"got {step_values.shape[0]}"
                )
            step_eig_buf.append(step_values)
            step_accept_buf.append(accepted)

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
        ev_buf.append(np.stack(eigs))
        if corrs is not None:
            corr_buf.append(corrs)

        if i % save_every == 0:
            _flush_buffers(ev_buf, corr_buf, paths)
            _flush_step_eig_buffers(step_eig_buf, step_accept_buf, step_eigs_path, step_eig_labels)
            if chunk is not None:
                chunk.flush()
            print(f"Iteration {i}, acceptance = {acc_count/i:.3f}, " + model.status_string())
            if save_checkpoints:
                model.save_state(paths["ckpt"])

    _flush_buffers(ev_buf, corr_buf, paths)
    _flush_step_eig_buffers(step_eig_buf, step_accept_buf, step_eigs_path, step_eig_labels)
    if chunk is not None:
        chunk.flush()

    if acc_count / max(niters, 1) < 0.5:
        print("WARNING: Acceptance rate is below 50%")

    _stop_profile(profiler)
    return model


__all__ = ["run", "_load_model_module"]
