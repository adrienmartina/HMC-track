#!/usr/bin/env python
"""CLI entry point for matrix-hmc-track."""

from __future__ import annotations

import datetime
import sys
import time

from matrix_hmc_track.cli import parse_args
from matrix_hmc_track.config import configure
from matrix_hmc_track.simulation import _load_model_module, run


def main() -> None:
    args = parse_args(sys.argv[1:])

    configure(
        device=args.device,
        precision=args.precision,
        threads=args.threads,
        interop_threads=args.interop_threads,
    )

    model_module = _load_model_module(args.model)
    if not hasattr(model_module, "build_model"):
        raise ValueError(f"Model module for '{args.model}' must define build_model(args)")
    model = model_module.build_model(args)

    start = time.time()
    print("STARTED:", datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"))

    run(
        model,
        niters=args.niters,
        step_size=args.step_size,
        nsteps=args.nsteps,
        output=args.data_path,
        name=args.name,
        save_every=args.save_every,
        save_checkpoints=args.save,
        save_matrices=args.saveAllMats,
        save_evals=args.save_evals,
        save_step_eigenvalues=args.save_step_eigenvalues,
        track_escape=args.track_escape,
        escape_descent_steps=args.escape_descent_steps,
        escape_descent_step_size=args.escape_descent_step_size,
        escape_validation_halvings=args.escape_validation_halvings,
        escape_grad_tol=args.escape_grad_tol,
        escape_trx2_atol=args.escape_trx2_atol,
        escape_trx2_rtol=args.escape_trx2_rtol,
        escape_persistence=args.escape_persistence,
        stop_on_escape=args.stop_on_escape,
        resume=args.resume and not args.fresh,
        force=args.force,
        seed=args.seed,
        profile=args.profile,
        dry_run=args.dry_run,
        quiet_trajectories=args.quiet_trajectories,
    )

    print("Runtime =", time.time() - start, "s")


if __name__ == "__main__":
    main()
