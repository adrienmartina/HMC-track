"""Shared configuration: device selection, dtypes, and host runtime knobs."""

from __future__ import annotations

import os
import torch

# Ensure Apple GPUs fall back cleanly to CPU for unsupported ops.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


ALLOW_TF32 = _parse_bool_env("IKKT_ALLOW_TF32", True)
CPU_NUM_THREADS = _parse_positive_int_env("IKKT_NUM_THREADS")
CPU_NUM_INTEROP_THREADS = _parse_positive_int_env("IKKT_NUM_INTEROP_THREADS")


def _real_dtype_for(complex_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if complex_dtype == torch.complex64 else torch.float64


# Default to double-precision complex unless overridden via CLI.
dtype = torch.complex128
real_dtype = _real_dtype_for(dtype)


def _enable_tf32() -> None:
    if not ALLOW_TF32:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")


def configure_threads(
    num_threads: int | None = None,
    num_interop_threads: int | None = None,
) -> tuple[int, int]:
    """Configure the number of CPU threads used by PyTorch.

    When arguments are ``None``, the values already stored in
    :data:`CPU_NUM_THREADS` / :data:`CPU_NUM_INTEROP_THREADS` (which are
    themselves read from the ``IKKT_NUM_THREADS`` / ``IKKT_NUM_INTEROP_THREADS``
    environment variables at import time) are used instead.

    Args:
        num_threads: Number of intra-op CPU threads.  Passed to
            :func:`torch.set_num_threads`.
        num_interop_threads: Number of inter-op CPU threads.  Passed to
            :func:`torch.set_num_interop_threads`.  Silently ignored if PyTorch
            has already started its thread pool.

    Returns:
        Tuple ``(intra_threads, inter_threads)`` reflecting the values that
        were actually applied.
    """
    global CPU_NUM_THREADS, CPU_NUM_INTEROP_THREADS

    resolved_threads = CPU_NUM_THREADS if num_threads is None else num_threads
    resolved_interop = CPU_NUM_INTEROP_THREADS if num_interop_threads is None else num_interop_threads

    if resolved_threads is not None:
        if resolved_threads < 1:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(resolved_threads)
    else:
        resolved_threads = torch.get_num_threads()

    if resolved_interop is not None:
        if resolved_interop < 1:
            raise ValueError("num_interop_threads must be positive")
        try:
            torch.set_num_interop_threads(resolved_interop)
        except RuntimeError:
            resolved_interop = torch.get_num_interop_threads()
    else:
        resolved_interop = torch.get_num_interop_threads()

    CPU_NUM_THREADS = resolved_threads
    CPU_NUM_INTEROP_THREADS = resolved_interop
    return resolved_threads, resolved_interop


def configure_device(target: str = "auto") -> torch.device:
    """Select the compute device.

    Args:
        target: ``'auto'`` picks GPU when available, ``'cpu'`` forces CPU,
                ``'gpu'`` requires CUDA (falls back to CPU with a warning).
    """
    dev: torch.device
    if target == "cpu":
        dev = torch.device("cpu")
        print("Using CPU.")
    elif target in ("gpu", "auto"):
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            _enable_tf32()
            print(f"Using CUDA device: {torch.cuda.get_device_name(dev)}")
        else:
            if target == "gpu":
                print("WARNING: --device gpu requested but CUDA is not available; falling back to CPU.")
            else:
                print("No GPU found, using CPU.")
            dev = torch.device("cpu")
    else:
        raise ValueError(f"--device must be 'cpu', 'gpu', or 'auto'; got {target!r}")

    global device
    device = dev
    return dev


def configure_dtype(precision: str = "complex64") -> torch.dtype:
    """Set the global floating-point precision.

    Args:
        precision: ``'complex64'`` (default, faster) or ``'complex128'`` (higher accuracy).
    """
    global dtype, real_dtype
    if precision == "complex64":
        dtype = torch.complex64
    elif precision == "complex128":
        dtype = torch.complex128
    else:
        raise ValueError(f"--precision must be 'complex64' or 'complex128'; got {precision!r}")
    real_dtype = _real_dtype_for(dtype)
    return dtype


# Default to CPU until configure_device is called.
device = torch.device("cpu")


def configure(
    device: str = "auto",
    precision: str = "complex64",
    threads: int | None = None,
    interop_threads: int | None = None,
) -> None:
    """One-shot convenience function to configure device, dtype, and threading.

    Calls :func:`configure_threads`, :func:`configure_device`, and
    :func:`configure_dtype` in that order.  Suitable for use at the top of a
    script or notebook before constructing any model.

    Args:
        device: Compute device — ``'auto'`` (GPU when available, else CPU),
            ``'cpu'``, or ``'gpu'`` (CUDA required).
        precision: Floating-point precision — ``'complex64'`` (single, faster)
            or ``'complex128'`` (double, more accurate).
        threads: Intra-op CPU thread count; ``None`` uses the environment
            variable ``IKKT_NUM_THREADS`` or PyTorch's default.
        interop_threads: Inter-op CPU thread count; ``None`` uses
            ``IKKT_NUM_INTEROP_THREADS`` or PyTorch's default.
    """
    configure_threads(threads, interop_threads)
    configure_device(device)
    configure_dtype(precision)


__all__ = [
    "ALLOW_TF32",
    "CPU_NUM_INTEROP_THREADS",
    "CPU_NUM_THREADS",
    "configure",
    "configure_threads",
    "configure_device",
    "configure_dtype",
    "device",
    "dtype",
    "real_dtype",
]
