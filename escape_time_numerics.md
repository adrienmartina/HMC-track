# Escape-time classifier: two numerical failure modes

The classical-descent escape classifier has two independent ways of going wrong:

1. **The precision floor** (§3–§9) — the descent never *finishes*, because
   floating-point resolution stops it before it reaches a critical point.
   Governed by `--escape-grad-tol` and `--precision`.
2. **Barrier jumping** (§10) — the descent finishes, converges, and reports the
   *wrong basin*, because a single step displaces the configuration further than
   its own magnitude. Governed by `--escape-descent-step-size`.

The second is the more dangerous: it produces confident, converged, entirely
false escapes. Fixing the first does not fix the second. See §11 for combined
settings.

## 1. The descent iteration

The classifier answers: *if the randomness were switched off, which basin does
this configuration flow to?* It does so by steepest descent on the classical
potential, with an Armijo backtracking line search:

$$Y_{k+1} \;=\; P\!\left[\,Y_k \;-\; \eta_k \,\nabla V(Y_k)\,\right]$$

where $P$ projects onto the constraint surface (Hermitian, traceless) and the
step length is chosen per iteration as

$$\eta_k = \eta_0\,2^{-m_k},
\qquad
m_k=\min\Big\{m\ge 0 \;:\; V\big(P[Y_k-\eta\nabla V]\big)\;\le\; V(Y_k)\;-\;c\,\eta\,\|\nabla V(Y_k)\|^2\Big\}$$

with $c=10^{-4}$, $\eta_0=10^{-2}$, and at most 25 halvings.

The norm is Frobenius over all matrices:

$$\|G\|^2=\sum_{\mu}\operatorname{tr}\big(G_\mu^\dagger G_\mu\big)
        =\sum_{\mu,a,b}\big|G_{\mu,ab}\big|^2$$

Convergence is declared when $\|\nabla V\|\le\tau$. **Nothing else sets the
`converged` flag** — running out of steps, or the line search failing, both
leave it false.

## 2. What sets the scale of the gradient

The bosonic potential of `pikkt4d_type2` is

$$V(X)=\frac{N}{g}\left[
-\tfrac12\sum_{\mu\nu}\operatorname{tr}\big[X_\mu,X_\nu\big]^2
\;+\;2i(1+\omega)\operatorname{tr}\big(X_1[X_2,X_3]\big)
\;+\;\sum_\mu c_\mu \operatorname{tr}X_\mu^2
\right]$$

with $c_{1,2,3}=\tfrac{\omega}{3}+\tfrac{2}{9}$ and $c_4=\tfrac{\omega}{3}$.

Everything sits behind the prefactor $N/g$. For $N=2$, $g=0.0059$:

$$\frac{N}{g}=339$$

The bracket is $O(1)$ when $\|X\|\sim1$, so $|V|\sim340$ — the measured value at
the stall was $V=-399.1$. Likewise

$$\|\nabla V\| \;\sim\; \frac{N}{g}\;O\!\left(\|X\|+\|X\|^2+\|X\|^3\right)$$

so gradients run in the hundreds; the measured opening value was
$\|\nabla V\|=914$. **The small coupling inflates both $V$ and $\nabla V$ by
~340.**

## 3. The precision floor

To first order, one step decreases the potential by

$$\Delta V \;=\; V\big(Y-\eta\nabla V\big)-V(Y)\;\approx\;-\,\eta\,\|\nabla V\|^{2}$$

But $V$ is stored with relative precision $\varepsilon$, so every evaluation
carries absolute noise

$$\delta V \;\approx\; \varepsilon\,|V|$$

The step is detectable only if $\eta\|\nabla V\|^2 > \varepsilon|V|$. Hence the
descent stalls once

$$\boxed{\;\|\nabla V\| \;\lesssim\; g_{\text{floor}}\;=\;\sqrt{\dfrac{\varepsilon\,|V|}{\eta}}\;}$$

Below this floor, $V(Y-\eta\nabla V)$ and $V(Y)$ are the *same floating-point
number*. The Armijo test degenerates to $V\le V$, which is true, so the step is
accepted — even though $Y-\eta\nabla V$ has already rounded back to $Y$. The
descent then walks in place until the step budget runs out.

**The tolerance must satisfy $\tau > g_{\text{floor}}$.** A $\tau$ below the
floor is not a strict criterion; it is an unreachable one.

### Null-step conditions, explicitly

The pathology needs two things simultaneously, both observed:

$$\eta\,\|\nabla V\|_\infty \;<\; \varepsilon\,\|Y\|_\infty
\qquad\text{(the update rounds away)}$$

$$c\,\eta\,\|\nabla V\|^2 \;<\; \varepsilon\,|V|
\qquad\text{(the required decrease rounds away)}$$

Measured at the freeze: $\eta=1.907\times10^{-8}$, $\|\nabla V\|=0.12$,
$|V|=399.1$, giving $\eta\|\nabla V\|\approx2.3\times10^{-9}$ against
$\varepsilon\|Y\|\approx1.2\times10^{-7}$, and
$c\,\eta\|\nabla V\|^2\approx2.8\times10^{-14}$ against
$\varepsilon|V|\approx4.8\times10^{-5}$. Both hold by wide margins.

## 4. Numbers vs. measurement

Using $|V|\approx400$ and the observed working step $\eta\approx6.25\times10^{-4}$:

| precision | $\varepsilon$ | predicted $g_{\text{floor}}$ | measured |
|---|---|---|---|
| `complex64`  | $1.19\times10^{-7}$  | $0.28$               | froze at $0.12$, $0.28$, $0.48$ |
| `complex128` | $2.22\times10^{-16}$ | $1.2\times10^{-5}$   | stalled at $1.06\times10^{-5}$; converged at $0.9\text{–}1.0\times10^{-5}$ |

The `complex128` prediction explains the tolerance sweep exactly:

| $\tau$ | vs. floor $1.2\times10^{-5}$ | result |
|---|---|---|
| $10^{-6}$ | below | **fails** (stalls at $1.06\times10^{-5}$) |
| $10^{-5}$ | just above | converges |
| $10^{-4}$ | comfortably above | converges |

It also explains the original run. With $\tau=10^{-1}$ in single precision, where
the floor is $0.28$, the tolerance sat *below the floor by a factor of 3* — so
any configuration needing a real journey was mathematically incapable of
satisfying it. The 18 classifications that did pass were configurations starting
essentially at the origin, and their reported gradients
($0.0885,\ 0.0975,\ 0.0933,\dots$) all pressed against the threshold from below.

## 5. Why double precision helps — and only by a square root

$$\frac{g_{\text{floor}}^{(64)}}{g_{\text{floor}}^{(32)}}
=\sqrt{\frac{\varepsilon_{64}}{\varepsilon_{32}}}
=\sqrt{\frac{2.22\times10^{-16}}{1.19\times10^{-7}}}
\approx 4.3\times10^{-5}$$

The floor drops by ~$23{,}000\times$. Note this is the *square root* of the
precision gain: a $10^{9}$ improvement in $\varepsilon$ buys only $\sim2\times10^{4}$
in gradient resolution, because $\Delta V\propto\|\nabla V\|^{2}$ near a critical
point. Curvature costs half your digits.

## 6. Why more steps cannot help

$g_{\text{floor}}$ depends on $\varepsilon$, $|V|$ and $\eta$ — and carries no
iteration index. It is a fixed barrier, not a slow approach. Gradient descent
would otherwise give $\|\nabla V_k\|\sim\rho^{k}$, but that decay terminates on
contact with the floor; past it, every further iteration is a null step.

Measured: identical results at $1000$, $5000$ and $20000$ steps.

## 7. A scale-free criterion

Since $V$ and $\nabla V$ both carry the $N/g$ prefactor, testing the raw
$\|\nabla V\|$ against a fixed $\tau$ makes the tolerance implicitly
coupling-dependent. The scale-free form would test the bracket's gradient:

$$\big\|\nabla F\big\| \;=\; \frac{g}{N}\,\big\|\nabla V\big\| \;\le\; \tau_{\text{rel}}$$

As written, scanning $g$ upward with fixed $\tau$ silently *tightens* the
criterion by $N/g$ at every point — worth correcting before comparing escape
times across couplings.

## 8. Settings for the precision floor alone

```
--precision complex128
--escape-grad-tol 1e-5
```

This clears the floor of §3: measured over 20 trajectories, 0 non-converged
classifications.

**These settings are not sufficient on their own.** They fix only whether the
descent finishes, not whether it finishes in the right basin — see §10. The
combined recommendation is in §11.

## 9. Choosing $\tau$ across a parameter scan

$\tau$ is `--escape-grad-tol`, tested directly against $\|\nabla V\|$. It is
**not** a universal constant: measured in `complex128`, the floor moves by
$\sim300\times$ over a modest range of $(N,g)$.

| $N$ | $g$ | $N/g$ | $\lvert V\rvert$ | $\eta_{\text{work}}$ | floor (measured) | $\sqrt{\varepsilon\lvert V\rvert/\eta}$ |
|---|---|---|---|---|---|---|
| 2 | 0.059  | 34   | 39.2 | $1.00\times10^{-2}$ | $2.62\times10^{-7}$ | $9.33\times10^{-7}$ |
| 2 | 0.0295 | 68   | 78.5 | $5.00\times10^{-3}$ | $1.25\times10^{-6}$ | $1.87\times10^{-6}$ |
| 2 | 0.0118 | 169  | 196  | $1.25\times10^{-3}$ | $3.07\times10^{-6}$ | $5.90\times10^{-6}$ |
| 2 | 0.0059 | 339  | 392  | $6.25\times10^{-4}$ | $1.06\times10^{-5}$ | $1.18\times10^{-5}$ |
| 3 | 0.0059 | 508  | 589  | $3.13\times10^{-4}$ | $2.10\times10^{-5}$ | $2.04\times10^{-5}$ |
| 4 | 0.0059 | 678  | 785  | $3.13\times10^{-4}$ | $4.44\times10^{-5}$ | $2.36\times10^{-5}$ |
| 6 | 0.0059 | 1017 | 5885 | $1.56\times10^{-4}$ | $7.59\times10^{-5}$ | $9.15\times10^{-5}$ |

(median over 3 seeds; the prediction uses the *working* step size, not the
collapsed $\eta$ at the stall.)

### Why there is no closed form

Two effects defeat a simple $\tau\propto N/g$ rule:

1. **$\eta$ saturates.** At $g=0.059$ the line search never backtracks, so
   $\eta=\eta_0$ and the floor falls below the scaling prediction. $\eta$ is set
   by curvature at runtime and cannot be folded into a formula.
2. **$\lvert V\rvert$ is not $\propto N/g$ across $N$.** From $N=2$ to $N=6$ at
   fixed $g$, $N/g$ grows $3\times$ but $\lvert V\rvert$ grows $15\times$,
   because the traces grow with matrix size.

Empirically the floor goes as $\sim(N/g)^{1.7}$ along the $g$ direction, but
that exponent is a one-decade fit with $\eta$ saturating at one end — not
something to extrapolate.

### The rule that holds

Normalized by the potential, the floor is nearly constant — spread of $8\times$
against $300\times$ for the raw value:

$$6.7\times10^{-9}\;\lesssim\;\frac{g_{\text{floor}}}{\lvert V\rvert}\;\lesssim\;5.7\times10^{-8}$$

Hence a practical setting with $40\text{–}150\times$ margin everywhere tested:

$$\boxed{\;\tau \;\approx\; 10^{-6}\,\lvert V\rvert\;}$$

For the current run $\lvert V\rvert\approx400$, giving $\tau\approx4\times10^{-4}$.
The potential is recorded per classification in `escape.npz`, so this rule is
self-calibrating rather than predictive.

### Rigorous calibration

At each new $(N,g)$: run ~20 trajectories with $\tau=10^{-30}$ so every descent
runs to its stall, read the `grad_norm` column from `escape.npz` — that
distribution *is* the floor at that parameter point — and set $\tau$ to
$10\times$ its 90th percentile.

This matters for a coupling scan: a $\tau$ that drifts relative to the floor
biases escape times differently at each $g$, which would contaminate any
trend read off the scan.

## 10. Barrier jumping: a converged descent in the wrong basin

Everything above concerns whether the descent *finishes*. This section concerns
whether it finishes in the *right place* — an independent failure that survives
every fix in §1–§9.

### Symptom

At $N=2$, $g=0.0059$, `complex128`, $\tau=10^{-4}$, $\eta_0=10^{-2}$ (default):
6 of 30 trajectories classified as escaped (iterations 7–11 and 15), each
landing at $\mathrm{Tr}X^2_{123}/N=2.0833$ with $V=-392.34$, **converged**, in
~40 steps. Yet the HMC casimir stayed in $[0.008,\,0.072]$ throughout — the
configurations never left the neighbourhood of the origin.

The status-line `casimir` and the classifier's basin observable are the *same*
quantity, $\operatorname{tr}(X_1^2+X_2^2+X_3^2)/N$. The difference is only where
it is evaluated: `casimir` on the HMC configuration, `trx2_flow` on the
descended one.

### The classification does not track the observable

| iteration | casimir (HMC) | classified |
|---|---|---|
| 11 | 0.02737 | escaped |
| 12 | **0.07235** | in-basin |
| 15 | 0.03479 | escaped |

Escaped range $[0.0277,\,0.0700]$; in-basin range $[0.0082,\,0.0724]$ — fully
overlapping, with the **largest** casimir of the run classified in-basin.

### Diagnosis: reduce $\eta_0$ and every escape disappears

Descending the saved configurations directly:

| iteration | casimir | $\eta_0=10^{-2}$ | $\eta_0=10^{-3}$ | $\eta_0=10^{-4}$ |
|---|---|---|---|---|
| 7  | 0.06995 | fuzzy sphere | origin | origin |
| 8  | 0.05136 | fuzzy sphere | origin | origin |
| 9  | 0.05591 | fuzzy sphere | origin | origin |
| 10 | 0.04083 | fuzzy sphere | origin | origin |
| 11 | 0.02737 | fuzzy sphere | origin | origin |
| 15 | 0.03479 | fuzzy sphere | origin | origin |

All 30 configurations belong to the origin's basin. **Every escape was an
artifact of the descent step size.**

$\mathrm{Tr}X^2_{123}/N=2.0833$ *is* a genuine critical point ($V=-392.34$, and
the descent converges there cleanly) — the configurations were simply never in
its basin.

### Mechanism

At iteration 7: $\|X\|_F=0.3875$, $\|\nabla V\|=57.1$, $V=+12.17$. The first
trial displacement is

$$\eta_0\,\|\nabla V\| \;=\; 10^{-2}\times 57.1 \;=\; 0.571 \;=\; 1.5\,\|X\|_F$$

The first step displaces the configuration by more than its own magnitude,
landing beyond the barrier. Armijo accepts it immediately — with no backtracking
at all — because

$$V_{\text{fuzzy}}=-392.34 \;\lll\; V_{\text{start}}=+12.17$$

**Armijo guarantees descent, not locality.** A line search designed to find *a*
minimum was used to decide *which* minimum, and those are different questions.

### The condition that avoids it

Near the origin the quadratic term dominates, $\nabla V\approx 2\frac{N}{g}\bar c\,X$
with $\bar c=\tfrac{\omega}{3}+\tfrac29$, so

$$\frac{\|\nabla V\|}{\|X\|}\;\approx\;2\bar c\,\frac{N}{g}\;=\;377
\qquad (N=2,\;g=0.0059,\;\omega=1)$$

Requiring the first displacement to stay below a fraction $\alpha$ of $\|X\|$:

$$\boxed{\;\eta_0 \;\lesssim\; \frac{\alpha\,g}{2\bar c\,N}\;\approx\;0.9\,\alpha\,\frac{g}{N}\;}$$

With $\alpha=0.05$: $\eta_0\lesssim1.3\times10^{-4}$.

**$\eta_0$ scales as $g/N$** — it must *shrink* as the coupling shrinks, in the
opposite direction to $\tau$, which must *grow*.

### Cost, and a correction to §6

At $\eta_0=10^{-4}$ the descent takes ~526 steps, and ~5300 at $10^{-5}$, versus
~43 at the default $10^{-2}$. **The step budget therefore does matter** once
$\eta_0$ is set correctly. Section 6 measured the budget to be irrelevant, but
did so only at the (too large) default $\eta_0$; that conclusion does not carry
over.

### The built-in guard

`--escape-validation-halvings` re-runs each escape at halved $\eta_0$ and
requires agreement. It defaults to $0$ — disabled. A single halving already
flips every false escape here:

| halvings | $\eta_0$ | $\mathrm{Tr}X^2_{123}/N$ | destination |
|---|---|---|---|
| 0 | $1.00\times10^{-2}$ | $2.0833$ | fuzzy sphere |
| 1 | $5.00\times10^{-3}$ | $2.9\times10^{-14}$ | origin |
| 2 | $2.50\times10^{-3}$ | $7.8\times10^{-25}$ | origin |
| 3 | $1.25\times10^{-3}$ | $7.9\times10^{-18}$ | origin |

Enabling it would have caught this immediately.

### Structural fix

The descent has no locality control at all. A trust region — capping each step
at a fixed fraction of $\|X\|$ — would make the classifier robust to $\eta_0$
instead of silently dependent on it, and would remove this failure mode rather
than requiring it to be tuned around.

## 11. Combined settings

```
--precision complex128
--escape-descent-step-size 1e-4
--escape-descent-steps 5000
--escape-grad-tol 1e-4
--escape-validation-halvings 2
```

Scaling across a parameter scan — the two knobs move in **opposite**
directions:

$$\tau \;\approx\; 10^{-6}\,|V| \quad\text{(grows as \(g\) shrinks)}
\qquad\qquad
\eta_0 \;\approx\; 0.05\,\frac{g}{N} \quad\text{(shrinks as \(g\) shrinks)}$$

### Checklist before trusting a run

- `escape_convergence_fraction` $=1.0$ in `run_stats.json`.
- Escapes survive the validation halvings.
- Spot-check: descend a flagged configuration at $\eta_0/10$ and confirm the
  same destination.
- Escapes should **persist** across consecutive trajectories. Flickering —
  escaped, then in-basin, then escaped — is the signature of §10, not of a
  physical transition.
- Compare the flagged iterations against the `casimir` trace. A configuration
  reported as escaped while its casimir sits at the ensemble average is a false
  positive.
