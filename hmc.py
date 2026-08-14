"""Model-agnostic Hybrid Monte Carlo kernels: leapfrog and Metropolis step."""

import math
import numpy as np
import random
from dataclasses import dataclass, replace
from typing import Any
import torch

from matrix_hmc_track.algebra import random_hermitian

@dataclass
class HMCParams:
    """Integrator parameters for a single HMC trajectory.

    Attributes:
        dt: Leapfrog step size.  Typically computed as ``step_size / nsteps``.
        nsteps: Number of leapfrog integration steps per trajectory.
    """
    dt: float
    nsteps: int
    

def hamil(X: torch.Tensor, mom_X: torch.Tensor, model: Any) -> float:
    """Compute the total HMC Hamiltonian ``H = V(X) + K(P)``.

    The kinetic term is ``K = (1/2) sum_j Tr(P_j^2)`` where the sum runs over
    all matrices in the configuration.

    Args:
        X: Current configuration tensor of shape ``(nmat, N, N)``.
        mom_X: Momenta tensor of the same shape as ``X``.
        model: A :class:`~matrix_hmc_track.models.base.MatrixModel` instance providing
            ``potential(X)`` and the matrix count ``nmat``.

    Returns:
        Total Hamiltonian as a Python float.
    """
    ham = model.potential(X).item()
    kin = 0.0
    for j in range(model.nmat):
        Pj = mom_X[j]
        kin = kin + 0.5 * torch.trace(Pj @ Pj).real
    return ham + kin.item()


def leapfrog(X: torch.Tensor, hmc_params: HMCParams, model: Any) -> tuple[torch.Tensor, float, float]:
    """Run a symplectic leapfrog trajectory from configuration *X*.

    Momenta are refreshed by drawing from :func:`~matrix_hmc_track.algebra.random_hermitian`
    at the start of each trajectory.  The integrator uses the velocity Verlet
    (leapfrog) scheme::

        P_{1/2} = P_0 - (dt/2) F(X_0)
        X_k     = X_{k-1} + dt P_{k-1/2}     for k = 1, ..., nsteps-1
        P_{k+1/2} = P_{k-1/2} - dt F(X_k)
        X_final = X_{n-1} + (dt/2) P_final

    If the model implements ``begin_trajectory(X)`` it is called before
    integrating; ``end_trajectory(accepted)`` is **not** called here (see
    :func:`update`).

    Args:
        X: Initial configuration of shape ``(nmat, N, N)``.
        hmc_params: Step size and number of steps.
        model: Model providing ``force(X)`` and matrix metadata.

    Returns:
        Tuple ``(X_new, H_init, H_final)`` where *X_new* is the proposed
        configuration and *H_init* / *H_final* are the Hamiltonians evaluated
        at the start and end of the trajectory.
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

    X = X + 0.5 * dt_local * mom_X

    for _ in range(1, hmc_params.nsteps):
        f_X = model.force(X)
        mom_X = mom_X - dt_local * f_X
        X = X + dt_local * mom_X

    f_X = model.force(X)
    mom_X = mom_X - dt_local * f_X
    X = X + 0.5 * dt_local * mom_X

    ham_final = hamil(X, mom_X, model)
    return X, ham_init, ham_final


def update(
    acc_count: int,
    hmc_params: HMCParams,
    model: Any,
    reject_prob: float = 1.0,
    verbose: bool = True,
):
    """Run one HMC trajectory and apply a Metropolis accept/reject step.

    The model's internal state is updated in-place: on acceptance ``model.set_state``
    receives the proposed configuration; on rejection the previous state is restored.
    Progress is printed to stdout for every step.

    Args:
        acc_count: Running accepted-step counter (incremented on acceptance).
        hmc_params: Leapfrog integrator parameters.
        model: A :class:`~matrix_hmc_track.models.base.MatrixModel` instance whose
            state will be mutated.
        reject_prob: Rescales the Metropolis probability by this factor so that
            ``p_accept = min(1, reject_prob * exp(-dH))``.  A value of 1.0
            (default) gives standard HMC.
    Returns:
        Updated acceptance counter ``acc_count``.
    """
    X = model.get_state()
    X_bak = X.clone()
    X_new, H0, H1 = leapfrog(X, hmc_params, model)
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
        if verbose:
            print(f"ACCEPT: dH={dH: 8.3f}, expDH={np.exp(-dH): 8.3f}, H0={H0: 8.4f}, ", model.status_string())
    else:
        model.set_state(X_bak)
        if verbose:
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

    return acc_count


def thermalize(model: Any, hmc_params: HMCParams, steps: int = 10) -> None:
    """Run short, highly-accepting trajectories to drive the system toward equilibrium.

    Uses a modified :class:`HMCParams` with ``2x nsteps`` and ``dt / 20`` so that
    the trajectories stay cheap and almost always accept, gradually moving away from
    the initial configuration without wasting many evaluations.

    Args:
        model: A :class:`~matrix_hmc_track.models.base.MatrixModel` instance.
        hmc_params: Base integrator parameters whose ``nsteps`` and ``dt`` are
            rescaled internally.
        steps: Number of thermalization trajectories to run (default: ``10``).
    """
    print("Thermalization steps, accept most jumps")
    therm_params = replace(hmc_params, nsteps=int(hmc_params.nsteps * 2), dt=hmc_params.dt / 20.0)
    acc_count = 0
    for _ in range(steps):
        acc_count = update(acc_count, therm_params, model)
    print("End of thermalization ", model.status_string())

__all__ = [
    "HMCParams",
    "hamil",
    "leapfrog",
    "update",
    "thermalize",
]
