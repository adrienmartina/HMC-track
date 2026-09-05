# matrix-hmc-track

A GPU-accelerated implementation of Hybrid Monte Carlo (HMC) for matrix models, with automatic fallback to CPU when a GPU is unavailable. Built on PyTorch, it provides automatic differentiation, efficient Pfaffian computations, and optimized linear algebra routines. The package includes a collection of built-in bosonic and fermionic models, as well as a straightforward interface for implementing custom models.

## Installation

```bash
pip install matrix-hmc-track
```

Requires Python ≥ 3.9, `torch`, and `numpy`.

---

## Quick start

### As a library

```python
import matrix_hmc_track as hmc
from matrix_hmc_track.models.pikkt10d import PIKKT10DModel

hmc.configure(device="auto", precision="complex64")

model = PIKKT10DModel(ncol=20, couplings=[0.1], spin=0, pfaffian_every=1)

hmc.run(
    model,
    niters=500,
    step_size=0.07,
    nsteps=150,
    output="data",
    name="triv",
)
```

### As a CLI

```bash
matrix-hmc-track --model pikkt10d --ncol 20 --coupling 0.1 \
           --step-size 0.07 --nsteps 150 --niters 500 \
           --name triv --data-path data --spin 0 --pfaffian-every 1
```

---

## Models

### `pikkt4d_type1` — Type I polarized IKKT, *SO*(4) invariant

$$S = \frac{N}{g}\mathrm{Tr}\Bigl[-\tfrac{1}{4}[X_I,X_J]^2 - \tfrac{i}{2}\bar\psi\,\Gamma^I[X_I,\psi] + X_I^2 + \eta\,\bar\psi\psi\Bigr]$$

- Fixed $D=4$, coupling `g`, optional deformation `eta` (default 1.0)
- Flags: `--eta`, `--massless` (drops $X^2$ and Myers terms)

```python
from matrix_hmc_track.models.pikkt4d_type1 import PIKKTTypeIModel

model = PIKKTTypeIModel(ncol=50, couplings=[150.0], eta=1.0)
model = PIKKTTypeIModel(ncol=30, couplings=[1.0], massless=True)
```

### `pikkt4d_type2` — Type II polarized IKKT, *SO*(3) invariant

$$S = \frac{N}{g}\mathrm{Tr}\Bigl[-\tfrac{1}{4}[X_I,X_J]^2 - \tfrac{i}{2}\bar\psi\,\Gamma^I[X_I,\psi] + \tfrac{i(2+2\omega)}{3}\epsilon_{ijk}X_iX_jX_k + \cdots\Bigr]$$

- Fixed $D=4$, couplings `g` and `omega`
- Flags: `--spin` (fuzzy-sphere initial condition), `--bosonic`, `--lorentzian`

```python
from matrix_hmc_track.models.pikkt4d_type2 import PIKKTTypeIIModel

model = PIKKTTypeIIModel(ncol=48, couplings=[0.1, 1.0], spin=0)
model = PIKKTTypeIIModel(ncol=48, couplings=[10.0, 1.0], spin=1)
```

### `pikkt10d` — 10D polarized IKKT

- Fixed $D=10$, single coupling `g`, $\Omega=1$
- Flags: `--massless`, `--pfaffian-every K` (measure Pfaffian every K trajectories), `--spin`

```python
from matrix_hmc_track.models.pikkt10d import PIKKT10DModel

model = PIKKT10DModel(ncol=20, couplings=[0.1], spin=0, pfaffian_every=1)
```

### `yangmills` — $D$-dimensional Yang-Mills

- Variable dimension, requires `--nmat D`
- Single coupling `g`, optional mass term `--mass`

```python
from matrix_hmc_track.models.yangmills import YangMillsModel

model = YangMillsModel(dim=4, ncol=50, couplings=[150.0])
```

### `adjoint_det` — Adjoint determinant model

- Variable dimension, requires `--nmat D`

```python
from matrix_hmc_track.models.adjoint_det import AdjointDetModel

model = AdjointDetModel(dim=6, ncol=40, couplings=[50.0])
```

### `susyym_3d` — 3D SUSY Yang-Mills with massive adjoint fermions

- Fixed $D=3$, flags: `--fermion-mass`, `--boson-mass`

```python
from matrix_hmc_track.models.susyym_3d import SUSYYM3DModel

model = SUSYYM3DModel(ncol=12, couplings=[60.0], fermion_mass=1.0)
```

### `1mm` — Single-matrix polynomial model

$$V(X) = \sum_n t_n \mathrm{Tr}(X^n)$$

- Couplings `t1 t2 ...` set the polynomial coefficients

```python
import importlib
mm = importlib.import_module("matrix_hmc_track.models.1mm")
model = mm.OneMatrixPolynomialModel(ncol=50, couplings=[1.0, -0.5])
```

---

## `hmc.run()` reference

```python
hmc.run(
    model,                    # MatrixModel instance
    niters=100,               # HMC trajectories
    step_size=0.5,            # total leapfrog length (dt = step_size / nsteps)
    nsteps=50,                # leapfrog steps per trajectory
    output="data",            # root directory for output files
    name="run",               # subdirectory prefix
    save_every=10,            # flush observables every K trajectories
    save_checkpoints=True,    # write checkpoint.pt every save_every steps
    save_matrices=False,      # also dump raw matrix snapshots
    resume=False,             # append to existing output, load checkpoint
    force=False,              # overwrite existing output files
    seed=None,                # RNG seed for reproducibility
    profile=False,            # print cProfile top-10 at the end
    dry_run=False,            # print config and return without running
)
```

## `hmc.configure()` reference

```python
hmc.configure(
    device="auto",        # "auto" | "cpu" | "gpu"
    precision="complex64",# "complex64" | "complex128"
    threads=None,         # PyTorch intra-op CPU threads
    interop_threads=None, # PyTorch inter-op CPU threads
)
```

---

## Adding a custom model

Create any `.py` file that defines a `MatrixModel` subclass and a `build_model(args)` factory:

```python
# my_model.py
from matrix_hmc_track.models.base import MatrixModel
import matrix_hmc_track.config as cfg
import torch

model_name = "my_model"

class MyModel(MatrixModel):
    def __init__(self, ncol, couplings):
        super().__init__(nmat=3, ncol=ncol)
        self.couplings = couplings
        self.g = couplings[0]
        self.is_hermitian = True
        self.is_traceless = True

    def load_fresh(self):
        self.set_state(torch.zeros(self.nmat, self.ncol, self.ncol,
                                   dtype=cfg.dtype, device=cfg.device))

    def potential(self, X=None):
        X = self._resolve_X(X)
        return self.g * torch.einsum("bij,bji->", X, X).real

    def measure_observables(self, X=None):
        X = self._resolve_X(X)
        eigs = [torch.linalg.eigvalsh(X[i]).real for i in range(self.nmat)]
        return eigs, None

def build_model(args):
    return MyModel(ncol=args.ncol, couplings=args.coupling)
```

### Use from Python

```python
import matrix_hmc_track as hmc
from my_model import MyModel

hmc.configure(device="cpu", precision="complex64")
model = MyModel(ncol=10, couplings=[1.0])
hmc.run(model, niters=100, step_size=0.3, nsteps=50, output="data", name="test")
```

### Use from the CLI

```bash
matrix-hmc-track --model ./my_model.py --ncol 10 --coupling 1.0 --niters 100
```

`--model` accepts a built-in name **or** a path to any `.py` file (relative or absolute).

---

## CLI reference

```bash
matrix-hmc-track --model <name|path>   # required
  --ncol N                       # matrix size
  --nmat D                       # number of matrices (variable-D models)
  --coupling g [omega ...]        # coupling constant(s)
  --niters K                     # HMC trajectories
  --step-size s --nsteps n       # leapfrog: dt = s/n
  --device auto|cpu|gpu
  --precision complex64|complex128
  --name NAME --data-path DIR    # output location
  --save / --no-save             # write checkpoint every --save-every steps
  --save-every K
  --saveAllMats                  # also dump raw matrix snapshots
  --resume                       # load checkpoint and append to existing output
  --fresh                        # ignore checkpoint even if --resume is set
  --force                        # overwrite existing output files
  --source "np.linspace(-1,1,N)" # external source term
  --seed S                       # RNG seed
  --threads T --interop-threads T
  --dry-run                      # print config, do not run
  --list-models                  # show built-in models
  --generate-config              # print sample YAML config for --model
  --config FILE                  # load YAML/TOML/JSON config (CLI overrides)
```

Model-specific flags:

| Flag | Models |
|------|--------|
| `--eta` | `pikkt4d_type1` |
| `--massless` | `pikkt4d_type1`, `pikkt10d` |
| `--spin` | `pikkt4d_type2`, `pikkt10d` |
| `--bosonic` | `pikkt4d_type2` |
| `--lorentzian` | `pikkt4d_type2` |
| `--pfaffian-every K` | `pikkt10d` |
| `--mass` | `yangmills` |
| `--fermion-mass`, `--boson-mass` | `susyym_3d` |
| `--det-coeff` | `adjoint_det` |

Environment variables:

| Variable | Effect |
|----------|--------|
| `IKKT_NUM_THREADS` | Default PyTorch intra-op thread count |
| `IKKT_NUM_INTEROP_THREADS` | Default PyTorch inter-op thread count |

---

## Outputs

Each run writes into `<data-path>/<name>_<model>_D<nmat>_N<ncol>/`:

| File | Contents |
|------|----------|
| `evals.npz` | Eigenvalue measurements (key `"values"`, shape `[niters, nmat, ncol]`) |
| `corrs.npz` | Correlator measurements (key `"values"`) |
| `metadata.json` | Model + run parameters |
| `checkpoint.pt` | Last saved configuration (if `--save`) |
| `all_mats/` | Chunked raw matrix snapshots (if `--saveAllMats`) |

Load observables:

```python
import numpy as np
d = np.load("data/triv_pikkt10d_D10_g0.1_N20/evals.npz")
evals = d["values"]  # shape (niters, 10, 20)
```

---

## Repository layout

```
matrix_hmc_track/
  __init__.py       run(), configure()
  simulation.py     run() implementation + I/O helpers
  main.py           CLI entry point (~50 lines)
  cli.py            argument parsing
  config.py         device / dtype / threading setup
  hmc.py            leapfrog integrator + Metropolis step
  algebra.py        Hermitian draws, adjoint maps, projections
  pfaffian.py       Pfaffian computation
  models/
    base.py         MatrixModel base class
    utils.py        shared model utilities
    pikkt4d_type1.py
    pikkt4d_type2.py
    pikkt10d.py
    yangmills.py
    adjoint_det.py
    susyym_3d.py
    1mm.py
tests/
  test_models.py    dry-run smoke test for every model
  test_codebase.py  algebra, HMC, and model unit tests
  test_cli.py       CLI argument parsing tests
  test_ad_matrix.py adjoint-matrix and Pfaffian tests
```
