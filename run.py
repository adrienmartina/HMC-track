from pathlib import Path
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # or remove this line if you have a display


def get_arg(cmd: list[str], flag: str, nvals: int = 1):
    idx = cmd.index(flag)
    values = cmd[idx + 1 : idx + 1 + nvals]
    return values[0] if nvals == 1 else values


def run_dir_for(cmd: list[str]) -> Path:
    model = get_arg(cmd, "--model")
    ncol = int(get_arg(cmd, "--ncol"))
    g_str, omega_str = get_arg(cmd, "--coupling", 2)
    name = get_arg(cmd, "--name")
    data_path = Path(get_arg(cmd, "--data-path"))
    return data_path / (
        f"{name}_{model}_g{round(float(g_str), 4)}_omega{round(float(omega_str), 4)}_N{ncol}"
    )


cmds = [
    [
        "python3",
        "main.py",
        "--model",
        "pikkt4d_type1",
        "--ncol",
        "10",
        "--niters",
        "500",
        "--coupling",
        "50",
        "--step-size",
        "0.1",
        "--nsteps",
        "300",
        "--name",
        "RunTypeI",
        "--data-path",
        "outputs",
        "--force",
    ],
]


plot_dir = Path("plotTrX2")
plot_dir.mkdir(parents=True, exist_ok=True)

for cmd in cmds:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    run_dir = run_dir_for(cmd)
    evals = np.load(run_dir / "evals.npz")["values"]

    # For Hermitian X_1, Tr(X_1^2) is the sum of squared eigenvalues.
    tr_x1_sq = np.sum(evals[:, 0, :].real ** 2, axis=1)

    plt.plot(tr_x1_sq, label=r"$\mathrm{Tr}(X_1^2)$")
    plt.xlabel("Monte Carlo iteration")
    plt.ylabel(r"$\mathrm{Tr}(X_1^2)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{run_dir.name}.png")
    plt.clf()
