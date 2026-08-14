"""CLI for the matrix model HMC driver.

Supports config files (YAML / TOML / JSON) as well as plain command-line flags.
Config-file values are treated as defaults; explicit CLI flags always override them.

Priority order (highest → lowest):
  1. Explicit CLI flags
  2. Values in --config file
  3. Per-model defaults (keyed on --model)
  4. Global argparse defaults

Quick-start examples::

    # run with all defaults for a model
    matrix-hmc-track --model yangmills

    # load a config file, override one value on the command line
    matrix-hmc-track --config myrun.yaml --niters 2000

    # print available models
    matrix-hmc-track --list-models

    # dump a sample YAML config for a model (stdout)
    matrix-hmc-track --model pikkt4d_type1 --generate-config
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_DATA_PATH = "data"
DEFAULT_PROFILE = False

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_DISCOVERY_EXCLUDES = {"base", "utils"}

# Models whose nmat is fixed (can be omitted on the CLI).
FIXED_NMAT_MODELS: dict[str, int] = {
    "pikkt4d_type1": 4,
    "pikkt4d_type2": 4,
    "pikkt4d_type3": 4,
    "pikkt10d": 10,
    "susyym_3d": 3,
}

# Per-model sensible starting defaults.  These are applied *before* the config
# file and CLI, so they are always overridable.
MODEL_DEFAULTS: dict[str, dict] = {
    "pikkt4d_type1": {
        "ncol": 10,
        "coupling": [0.2],
        "step_size": 0.5,
        "nsteps": 50,
    },
    "pikkt4d_type2": {
        "ncol": 10,
        "coupling": [0.5, 0.5],
        "step_size": 0.5,
        "nsteps": 50,
    },
    "pikkt4d_type3": {
        "ncol": 10,
        "coupling": [0.5, 0.5],
        "step_size": 0.5,
        "nsteps": 50,
    },
    "pikkt10d": {
        "ncol": 10,
        "coupling": [1.0],
        "step_size": 0.3,
        "nsteps": 50,
    },
    "susyym_3d": {
        "ncol": 10,
        "coupling": [1.0],
        "step_size": 0.1,
        "nsteps": 50,
    },
    "yangmills": {
        "ncol": 50,
        "nmat": 4,
        "coupling": [1.0],
        "step_size": 0.5,
        "nsteps": 50,
    },
    "adjoint_det": {
        "ncol": 50,
        "nmat": 4,
        "coupling": [1.0],
        "step_size": 0.1,
        "nsteps": 50,
    },
    "1mm": {
        "ncol": 50,
        "coupling": [1.0, -0.5],
        "step_size": 0.5,
        "nsteps": 50,
    },
}

def _discover_known_models() -> list[str]:
    if not _MODEL_DIR.is_dir():
        return []
    return sorted(
        p.stem
        for p in _MODEL_DIR.glob("*.py")
        if not p.stem.startswith("_") and p.stem not in _MODEL_DISCOVERY_EXCLUDES
    )


_KNOWN_MODELS = _discover_known_models()


# --------------------------------------------------------------------------- #
# Config-file loading
# --------------------------------------------------------------------------- #

def _load_config_file(path: str) -> dict:
    """Load YAML / TOML / JSON config file and return a plain dict.

    Keys are normalized: hyphens replaced with underscores so that
    ``step-size`` and ``step_size`` are treated identically.
    """
    ext = Path(path).suffix.lower()
    try:
        if ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError:
                raise ImportError(
                    "pyyaml is required for YAML config files.\n"
                    "  pip install pyyaml\n"
                    "Or use a .toml or .json config file instead."
                )
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

        elif ext == ".toml":
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    raise ImportError(
                        "Python 3.11+ tomllib or the tomli package is needed for TOML.\n"
                        "  pip install tomli\n"
                        "Or use a .yaml or .json config file instead."
                    )
            with open(path, "rb") as fh:
                data = tomllib.load(fh)

        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

        else:
            raise ValueError(
                f"Unrecognized config file extension {ext!r}. Use .yaml, .toml, or .json."
            )
    except (OSError, IOError) as exc:
        raise SystemExit(f"Cannot open config file {path!r}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Config file {path!r} must be a key-value mapping at the top level.")

    # Normalize keys.
    return {k.replace("-", "_"): v for k, v in data.items()}


def _normalize_config(cfg: dict) -> dict:
    """Coerce config-file values to the types argparse expects and handle
    renamed flags from older versions of the CLI."""
    out = dict(cfg)

    # Coupling must always be a list (users might write ``coupling: 1.0``).
    if "coupling" in out and not isinstance(out["coupling"], list):
        out["coupling"] = [out["coupling"]]

    # Back-compat: ``no_gpu: true`` → ``device: cpu``
    if "no_gpu" in out:
        if out.pop("no_gpu"):
            out.setdefault("device", "cpu")

    # Back-compat: ``complex64: true/false`` → ``precision: complex64/complex128``
    if "complex64" in out:
        val = out.pop("complex64")
        out.setdefault("precision", "complex64" if val else "complex128")

    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_source(expr: str):
    """Evaluate a numpy expression like ``np.linspace(-1, 1, 20)`` for --source."""
    try:
        return eval(expr, {"np": np}, {})  # noqa: S307 – controlled namespace
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --source expression {expr!r}: {exc}"
        ) from exc


def _default_nmat_for_model(model_lower: str) -> int | None:
    return FIXED_NMAT_MODELS.get(model_lower)


def _validate_source_shape(source: np.ndarray, ncol: int, expected_nmat: int | None) -> None:
    s = np.asarray(source)
    if s.ndim == 1:
        if s.shape != (ncol,):
            raise ValueError(
                f"--source 1D array must have shape ({ncol},), got {s.shape}"
            )
        return
    if s.ndim != 3:
        raise ValueError(
            f"--source must be shape ({ncol},) or (nmat, {ncol}, {ncol}), got {s.shape}"
        )
    if s.shape[1:] != (ncol, ncol):
        raise ValueError(
            f"--source 3D array must have shape (nmat, {ncol}, {ncol}), got {s.shape}"
        )
    if expected_nmat is not None and s.shape[0] != expected_nmat:
        raise ValueError(
            f"--source first dimension must equal nmat={expected_nmat}, got {s.shape[0]}"
        )


def _print_model_list() -> None:
    print("Available models:\n")
    print(f"  {'-'*28}")
    for name in _KNOWN_MODELS:
        print(f"  {name:<28}")
    print()


def _generate_config_yaml(model_name: str) -> str:
    """Return a sample YAML config string for a given model."""
    d = MODEL_DEFAULTS.get(model_name, {})
    nmat = FIXED_NMAT_MODELS.get(model_name)
    lines = [
        f"# Sample config for --model {model_name}",
        f"# Edit values as needed, then run:",
        f"#   matrix-hmc-track --config this_file.yaml",
        "",
        f"model: {model_name}",
        f"ncol: {d.get('ncol', 10)}",
    ]
    if nmat is not None:
        lines.append(f"# nmat is fixed at {nmat} for this model")
    elif "nmat" in d:
        lines.append(f"nmat: {d['nmat']}")
    coupling = d.get("coupling", [1.0])
    if len(coupling) == 1:
        lines.append(f"coupling: [{coupling[0]}]")
    else:
        lines.append(f"coupling: {coupling}")
    lines += [
        f"step_size: {d.get('step_size', 0.1)}",
        f"nsteps: {d.get('nsteps', 50)}",
        "niters: 100",
        "device: auto",
        "precision: complex64",
        'name: run',
        f"data_path: data",
        "save: true",
        "save_every: 10",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Parser construction
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- meta flags ----
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="Path to a YAML / TOML / JSON config file. CLI flags override file values.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        dest="list_models",
        help="Print available models with their defaults and exit.",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        dest="generate_config",
        help="Print a sample YAML config for --model to stdout and exit.",
    )

    # ---- core ----
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Built-in model name (run --list-models) or path to a .py file "
            "that defines build_model(args)."
        ),
    )
    parser.add_argument("--ncol", type=int, default=None,
                        help="Matrix size N (model default used if omitted).")
    parser.add_argument("--nmat", type=int, default=None,
                        help="Number of matrices D. Required for variable-D models (yangmills, adjoint_det).")
    parser.add_argument("--coupling", type=float, nargs="+", default=None,
                        help="Coupling constant(s). Number required depends on model (model default used if omitted).")
    parser.add_argument("--niters", type=int, default=100,
                        help="Number of HMC trajectories to run.")

    # ---- integrator ----
    parser.add_argument("--step-size", type=float, dest="step_size", default=None,
                        help="Total leapfrog trajectory length Δt (model default if omitted).")
    parser.add_argument("--nsteps", type=int, default=None,
                        help="Leapfrog steps per trajectory (model default if omitted).")

    # ---- device / precision ----
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help=(
            "Compute device. 'auto' picks GPU when available, "
            "'cpu' forces CPU, 'gpu' requires CUDA."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=["complex64", "complex128"],
        default="complex64",
        help="Floating-point precision. complex64 is faster; complex128 for higher accuracy.",
    )
    parser.add_argument(
        "--complex64",
        dest="precision",
        action="store_const",
        const="complex64",
        help="Backward-compatible alias for --precision complex64.",
    )

    # ---- I/O ----
    parser.add_argument("--name", type=str, default="run",
                        help="Prefix added to output directory name.")
    parser.add_argument("--data-path", type=str, dest="data_path", default=DEFAULT_DATA_PATH,
                        help="Root directory for output files and checkpoints.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True,
                        help="Write checkpoint every --save-every trajectories (default: on).")
    parser.add_argument("--save-every", type=int, dest="save_every", default=10,
                        help="Flush observables and (if --save) write checkpoint every K trajectories.")
    parser.add_argument("--saveAllMats", action="store_true",
                        help="Also dump raw matrix snapshots every --save-every trajectories.")
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing checkpoint and output files instead of starting fresh.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing checkpoint and start from zero fields.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output files in the selected run directory; fresh runs use a new _runN directory if needed.")
    parser.add_argument("--source", type=_parse_source, default=None,
                        metavar="EXPR",
                        help="Numpy expression for an external source, e.g. 'np.linspace(-1,1,20)'.")

    # ---- misc ----
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility.")
    parser.add_argument("--threads", type=int, default=None,
                        help="PyTorch intra-op CPU thread count (default: let PyTorch decide).")
    parser.add_argument("--interop-threads", type=int, dest="interop_threads", default=None,
                        help="PyTorch inter-op CPU thread count.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved configuration and exit without running.")
    parser.add_argument("--profile", dest="profile", action="store_true", default=DEFAULT_PROFILE,
                        help="Enable cProfile sampling.")
    parser.add_argument("--no-profile", dest="profile", action="store_false",
                        help="Disable cProfile sampling.")

    # ---- model-specific groups ----
    ym = parser.add_argument_group("Yang-Mills  (--model yangmills)")
    ym.add_argument("--mass", type=float, default=1.0,
                    help="Coefficient of the Tr(X_i²) mass term (1.0 = standard Yang-Mills).")

    s3d = parser.add_argument_group("3D SUSY YM  (--model susyym_3d)")
    s3d.add_argument("--fermion-mass", type=float, default=1.0, dest="fermion_mass",
                     help="Adjoint fermion mass deformation.")
    s3d.add_argument("--boson-mass", type=float, default=1.0, dest="boson_mass",
                     help="Boson Tr(X_i²) mass coefficient.")

    t1 = parser.add_argument_group("Type I  (--model pikkt4d_type1)")
    t1.add_argument("--eta", type=float, default=1.0,
                    help="Fermion deformation parameter η (1.0 = undeformed SUSY).")
    t1.add_argument("--massless", action="store_true",
                    help="Remove mass / Myers term (bare IKKT / Pfaffian).")

    t2 = parser.add_argument_group("Type II / 10D  (--model pikkt4d_type2 or pikkt4d_type3 or pikkt10d)")
    t2.add_argument("--spin", type=float, default=None,
                    help="Spin j for fuzzy-sphere background in X₁, X₂, X₃.")
    t2.add_argument("--bosonic", action="store_true",
                    help="Disable fermionic determinant (purely bosonic run).")
    t2.add_argument("--pfaffian-every", type=int, default=1, dest="pfaffian_every",
                    help="Measure Pfaffian observables every K trajectories.")
    t2.add_argument("--lorentzian", action="store_true",
                    help="Wick-rotate X₄ → i X₄ in the Type II potential.")

    ad = parser.add_argument_group("Adjoint det  (--model adjoint_det)")
    ad.add_argument("--det-coeff", type=float, default=0.5, dest="det_coeff",
                    help="Coefficient in the fermion determinant term.")

    return parser


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse arguments, merging config file and model defaults with CLI flags."""

    # ---- handle --list-models immediately (no model required) ----
    if "--list-models" in argv:
        _print_model_list()
        sys.exit(0)

    # ---- pass 1: extract --config and --model without full parsing ----
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--model", default=None)
    pre.add_argument("--generate-config", action="store_true", dest="generate_config", default=False)
    pre_args, _ = pre.parse_known_args(argv)

    # ---- load config file ----
    file_cfg: dict = {}
    if pre_args.config:
        file_cfg = _normalize_config(_load_config_file(pre_args.config))

    # ---- resolve model name (CLI > config file) ----
    model_name = (pre_args.model or file_cfg.get("model") or "").strip().lower()

    # ---- handle --generate-config ----
    if pre_args.generate_config:
        if not model_name:
            sys.exit("--generate-config requires --model <name>")
        if model_name not in _KNOWN_MODELS:
            sys.exit(
                f"Unknown model {model_name!r}. Run --list-models to see available models."
            )
        print(_generate_config_yaml(model_name), end="")
        sys.exit(0)

    # ---- merge defaults: model defaults < config file (config file wins) ----
    merged: dict = {}
    merged.update(MODEL_DEFAULTS.get(model_name, {}))
    merged.update(file_cfg)

    # ---- full parse with merged defaults applied ----
    parser = _build_parser()
    if merged:
        parser.set_defaults(**merged)
    args = parser.parse_args(argv)

    # handle --list-models / --generate-config that slipped through set_defaults
    if getattr(args, "list_models", False):
        _print_model_list()
        sys.exit(0)
    if getattr(args, "generate_config", False):
        if not args.model:
            parser.error("--generate-config requires --model <name>")
        if args.model.strip().lower() not in _KNOWN_MODELS:
            parser.error(f"Unknown model {args.model!r}. Run --list-models to see available models.")
        print(_generate_config_yaml(args.model.strip().lower()), end="")
        sys.exit(0)

    if args.model is None:
        parser.error(
            "--model is required (or set 'model:' in a --config file).\n"
            "Run --list-models to see available built-in models, "
            "or pass a path: --model ./my_model.py"
        )
    args.model = args.model.strip()
    # Only lowercase for named models (preserve case in file paths)
    is_path = "/" in args.model or args.model.endswith(".py")
    if not is_path:
        args.model = args.model.lower()

    # ---- apply per-model fallbacks for args that may still be None ----
    fallback = MODEL_DEFAULTS.get(args.model, {})
    if args.ncol is None:
        args.ncol = fallback.get("ncol", 10)
    if args.step_size is None:
        args.step_size = fallback.get("step_size", 0.5)
    if args.nsteps is None:
        args.nsteps = fallback.get("nsteps", 20)
    if args.coupling is None:
        args.coupling = list(fallback.get("coupling", [1.0]))

    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Raise ValueError / argparse errors for obviously invalid inputs."""
    if args.ncol < 1:
        raise ValueError("--ncol must be a positive integer")
    if args.niters < 1:
        raise ValueError("--niters must be a positive integer")
    if not args.coupling:
        raise ValueError("--coupling requires at least one value")

    model_val = args.model
    # Skip model-specific validation for file-path models
    if "/" in model_val or model_val.endswith(".py"):
        _validate_common_args(args)
        return

    model_lower = model_val.lower()

    if model_lower == "1mm":
        if len(args.coupling) < 1:
            raise ValueError("1mm model requires at least one coupling via --coupling t1 [t2 ...]")

    if model_lower == "pikkt4d_type1":
        if len(args.coupling) != 1:
            raise ValueError("pikkt4d_type1 requires exactly one coupling: --coupling g")
        if not math.isfinite(args.eta) or args.eta == 0.0:
            raise ValueError("--eta must be finite and non-zero")

    if model_lower == "pikkt4d_type2":
        if len(args.coupling) != 2:
            raise ValueError("pikkt4d_type2 requires exactly two couplings: --coupling g omega")
        
    if model_lower == "pikkt4d_type3":
        if len(args.coupling) != 2:
            raise ValueError("pikkt4d_type3 requires exactly two couplings: --coupling g omega")

    if model_lower == "pikkt10d":
        if len(args.coupling) != 1:
            raise ValueError("pikkt10d requires exactly one coupling: --coupling g")
        if args.nmat is not None and args.nmat != 10:
            raise ValueError("pikkt10d has fixed D=10; omit --nmat or set --nmat 10")

    if model_lower == "yangmills":
        if len(args.coupling) != 1:
            raise ValueError("yangmills requires exactly one coupling: --coupling g")
        if args.nmat is None or args.nmat < 2:
            raise ValueError("yangmills requires --nmat >= 2 (number of matrices D)")
        if not math.isfinite(args.mass):
            raise ValueError("--mass must be finite")

    if model_lower == "susyym_3d":
        if len(args.coupling) != 1:
            raise ValueError("susyym_3d requires exactly one coupling: --coupling g")
        if args.nmat is not None and args.nmat != 3:
            raise ValueError("susyym_3d has fixed D=3; omit --nmat or set --nmat 3")
        if not math.isfinite(args.fermion_mass):
            raise ValueError("--fermion-mass must be finite")
        if not math.isfinite(args.boson_mass):
            raise ValueError("--boson-mass must be finite")

    if model_lower == "adjoint_det":
        if len(args.coupling) != 1:
            raise ValueError("adjoint_det requires exactly one coupling: --coupling g")
        if args.nmat is None or args.nmat < 1:
            raise ValueError("adjoint_det requires --nmat >= 1")

    if args.nsteps < 1:
        raise ValueError("--nsteps must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.save_every < 1:
        raise ValueError("--save-every must be positive")
    if args.pfaffian_every < 1:
        raise ValueError("--pfaffian-every must be positive")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.interop_threads is not None and args.interop_threads < 1:
        raise ValueError("--interop-threads must be positive")
    if args.source is not None:
        expected_nmat = args.nmat if args.nmat is not None else _default_nmat_for_model(model_lower)
        _validate_source_shape(args.source, args.ncol, expected_nmat)


def _validate_common_args(args: argparse.Namespace) -> None:
    """Validations that apply regardless of model (used for file-path models)."""
    if args.nsteps < 1:
        raise ValueError("--nsteps must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.save_every < 1:
        raise ValueError("--save-every must be positive")
    if args.pfaffian_every < 1:
        raise ValueError("--pfaffian-every must be positive")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.interop_threads is not None and args.interop_threads < 1:
        raise ValueError("--interop-threads must be positive")
    if args.source is not None:
        _validate_source_shape(args.source, args.ncol, args.nmat)


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_PROFILE",
    "MODEL_DEFAULTS",
    "_KNOWN_MODELS",
    "parse_args",
    "validate_args",
]
