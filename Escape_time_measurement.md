# Escape-time measurement: definitions and procedure

How the autocorrelation time $\tau_{\rm int}$ is measured, how the mean escape time
$\bar T$ is estimated in the presence of censoring, and how the two are combined into
the normalized (dimensionless) escape time $\bar T/\tau_{\rm int}$ that is fitted
against $1/g$.

Reference implementation: `analyze_robust_ell1_precision.py`
(helpers in `metastability_analysis.py`; data written by `simulation.py`).
Reference dataset: `metastability_results_robust_ell1_precision5/`.

---

## 0. What a folder of runs must contain

One directory per replica. Each replica directory must have:

| file | key / field | meaning |
|---|---|---|
| `evals.npz` | `values`, shape `(niters, 7, ncol)` | per-trajectory eigenvalues; channels `0..3` are the Hermitian spectra of $X_1..X_4$ |
| `escape.npz` | `iteration`, `escaped`, `in_initial_basin`, `converged`, `trx2_flow`, `reference_trx2`, `tolerance_atol`, `tolerance_rtol` | per-trajectory basin classification |
| `run_stats.json` | `g`, `seed`, `nsteps`, `step_size`, `completed_trajectories`, `niters_requested`, `escape_censored`, `escape_first_reliable_iteration`, `escape_classification_reliable`, `escape_convergence_fraction`, `acceptance_rate` | per-run summary |

These are produced by a run launched with

```
--track-escape --stop-on-escape --fresh
```

`--stop-on-escape` is **not optional**: the estimator in §3 assumes each replica is
terminated at its first escape (see the warning there).

A replica is one independent measurement. Escape is irreversible — the chain does not
return to the metastable basin — so **one replica yields at most one escape event**.

---

## 1. Notation

Time is measured in HMC trajectories, $t = 1,2,\dots$. Replicas are indexed
$r = 1,\dots,R$ at fixed coupling $g$.

The scalar observable used for the autocorrelation is

$$O_t \;=\; R^2_{\rm all}(t)\;=\;\sum_{I=1}^{4}\operatorname{Tr}X_I^{\,2}(t)
\;=\;\sum_{I=1}^{4}\sum_{a=1}^{N}\lambda_{I,a}(t)^2 ,$$

read from channels `0..3` of `evals.npz` (`load_R2_all` in `metastability_analysis.py`).

The escape classifier uses a different, closely related scalar: gradient descent on the
classical potential is run from each configuration, and the descended point is labelled by

$$\rho_{\rm flow} \;=\; \frac{1}{N}\sum_{I=1}^{3}\operatorname{Tr}X_I^{\,2}\Big|_{\rm descended} .$$

The configuration is *in the initial basin* iff

$$\big|\rho_{\rm flow}-\rho_{\rm ref}\big| \;\le\; {\tt atol} + {\tt rtol}\,|\rho_{\rm ref}| ,$$

with $\rho_{\rm ref}$ obtained by descending the initial configuration. Otherwise it is
*escaped*. (In the reference dataset $\rho_{\rm ref}=0$, in-basin values are $\le 10^{-19}$,
and escaped values are $25/12 = 2.0833$ — the classification is unambiguous by
fourteen orders of magnitude.)

---

## 2. Replica selection

A replica enters the analysis only if all three hold:

1. **Complete.** Either it escaped, or it ran the full requested `niters_requested`
   (a killed run would otherwise be counted as a short censored observation and bias
   $\bar T$ downwards).
2. **Reliable.** `escape_classification_reliable == True`.
3. **Fully converged descents.** `escape_convergence_fraction >= 0.999999`.

Replicas failing any of these are recorded in `replicas.csv` with
`valid_for_analysis = False` and excluded from every quantity below. Matching of run
parameters (model, $N$, $\omega$, spin, `nsteps`, `step_size`, descent settings) is done
in `run_is_robust_ell1`; adapt that predicate for a new folder.

---

## 3. Autocorrelation time $\tau_{\rm int}$

### 3.1 Definition

For a stationary series with autocovariance
$C(k)=\big\langle (O_t-\langle O\rangle)(O_{t+k}-\langle O\rangle)\big\rangle$ and
$\rho(k)=C(k)/C(0)$,

$$\boxed{\;\tau_{\rm int}\;=\;\frac12\sum_{k=-\infty}^{\infty}\rho(k)
\;=\;\frac12+\sum_{k=1}^{\infty}\rho(k)\;}$$

Its meaning is fixed by the variance of a sample mean over $n$ correlated samples,

$$\operatorname{Var}(\bar O)\;=\;\frac{2\tau_{\rm int}}{n}\operatorname{Var}(O),
\qquad n_{\rm eff}=\frac{n}{2\tau_{\rm int}} .$$

Uncorrelated data gives $\tau_{\rm int}=\tfrac12$. **One statistically independent
configuration costs $2\tau_{\rm int}$ trajectories.** For a single exponential mode
$\rho(k)=e^{-k/\tau_{\rm exp}}$ one has
$\tau_{\rm int}=\tfrac12\coth\!\big(1/2\tau_{\rm exp}\big)$.

### 3.2 Which series

$\tau_{\rm int}$ must describe relaxation **inside** the metastable basin, not the global
relaxation of the chain — the slowest global mode *is* the escape, so normalizing by it
would be circular. Therefore, per replica, take

$$\text{segment } r \;=\; \big\{\,O_t \;:\; \texttt{in\_initial\_basin}_t \ \wedge\ \texttt{converged}_t \,\big\}$$

(`pre_escape_segment`). In practice this drops exactly one point per replica, the escape
trajectory itself.

### 3.3 Pooled estimator

Estimates are pooled over all $R$ replicas at fixed $g$ (segment $r$ has length $n_r$):

$$\bar O=\frac{\sum_r\sum_t O^{(r)}_t}{\sum_r n_r},
\qquad
\hat C(k)=\frac{\displaystyle\sum_{r=1}^{R}\ \sum_{t=1}^{n_r-k}\big(O^{(r)}_t-\bar O\big)\big(O^{(r)}_{t+k}-\bar O\big)}
{\displaystyle\sum_{r=1}^{R}\,(n_r-k)},
\qquad
\hat\rho(k)=\frac{\hat C(k)}{\hat C(0)} .$$

Two deliberate choices:

* **One common mean $\bar O$ across replicas.** Subtracting per-replica means would remove
  precisely the slow fluctuations being measured and bias $\tau_{\rm int}$ low.
* **Pair-count normalization $\sum_r (n_r-k)$**, not $\sum_r n_r$.

### 3.4 Windowing

The sum must be truncated: $\hat\rho(k)$ carries roughly constant noise at every lag while
the signal decays.

$$\hat\tau_{\rm int}(W)=\frac12+\sum_{k=1}^{W}\hat\rho(k),
\qquad
\text{bias}\simeq-\!\!\sum_{k>W}\rho(k),
\qquad
\operatorname{Var}\big[\hat\tau_{\rm int}(W)\big]\simeq\frac{2(2W+1)}{N_{\rm tot}}\,\tau_{\rm int}^{2},$$

with $N_{\rm tot}=\sum_r n_r$. The window is fixed by **Sokal's rule**, capped at the first
non-positive lag:

$$W=\min\Big\{\,W\ \ge 1\ :\ W\ \ge\ c\,\hat\tau_{\rm int}(W)\,\Big\},\qquad c=5 .$$

The residual truncation bias is then $O(e^{-c})\approx 0.7\%$. Finally

$$\tau_{\rm int}=\max\!\Big(\tfrac12,\ \hat\tau_{\rm int}(W)\Big).$$

### 3.5 Worked example (reference dataset, $g=0.0052$)

$R=461$ segments, $N_{\rm tot}=1{,}033{,}077$ samples:

| $W$ | $\hat\rho(W)$ | $\hat\tau_{\rm int}(W)$ | $5\,\hat\tau_{\rm int}(W)$ | $W \ge 5\hat\tau$ ? |
|---|---|---|---|---|
| 1 | 0.27168 | 0.77168 | 3.858 | no |
| 2 | 0.10562 | 0.87730 | 4.387 | no |
| 3 | 0.04639 | 0.92370 | 4.618 | no |
| 4 | 0.02309 | 0.94679 | 4.734 | no |
| **5** | **0.01093** | **0.95772** | **4.789** | **yes → stop** |

$$\tau_{\rm int}=0.957724 .$$

Note $\hat\rho(2)=0.106 > \hat\rho(1)^2=0.074$: the decay is not a single exponential,
which is why the sum is windowed rather than fitted.

---

## 4. Mean escape time $\bar T$ (right-censored)

### 4.1 Per-replica observation

For replica $r$ define the **exposure** $T_r$ and the **event indicator** $\delta_r$:

$$T_r=\texttt{completed\_trajectories}_r,
\qquad
\delta_r=\begin{cases}
1, & \text{replica escaped (uncensored)}\\
0, & \text{replica reached } n_{\rm iters} \text{ without escaping (censored)}
\end{cases}$$

For an escaped replica, $T_r$ *is* the first-passage time, because `--stop-on-escape`
terminates the run at the escape trajectory. For a censored replica, $T_r=n_{\rm iters}$
and the true escape time is only known to exceed it.

> **Warning.** Without `--stop-on-escape`, $T_r > T_r^{\rm escape}$ for escaped replicas and
> the estimator below silently inflates. Verify `observed_time == T_escape` for every
> uncensored replica before trusting the result.

### 4.2 The estimator

First-passage over a high barrier is a memoryless (Poisson) process, so
$T\sim\text{Exp}(1/\bar T)$ with density $f(t)=\bar T^{-1}e^{-t/\bar T}$ and survival
$S(t)=e^{-t/\bar T}$. The likelihood over replicas is

$$\mathcal{L}(\bar T)=\prod_{r=1}^{R} f(T_r)^{\delta_r}\,S(T_r)^{1-\delta_r}
=\prod_{r}\Big(\tfrac1{\bar T}\Big)^{\delta_r} e^{-T_r/\bar T},$$

$$\log\mathcal{L}=-\Big(\sum_r \delta_r\Big)\log\bar T-\frac{1}{\bar T}\sum_r T_r ,$$

and $\partial_{\bar T}\log\mathcal{L}=0$ gives the maximum-likelihood estimate

$$\boxed{\;\bar T \;=\; \frac{\displaystyle\sum_{r=1}^{R} T_r}{\displaystyle\sum_{r=1}^{R}\delta_r}
\;=\;\frac{\text{total exposure in trajectories}}{\text{number of escape events}}\;}$$

This is the ratio in which the **number of escape events** appears: it is the denominator
of the mean escape time. Censored replicas contribute their full exposure to the numerator
but nothing to the denominator, which is exactly right and is why runs that never escape
are still informative and must not be discarded.

### 4.3 Precision, and why it cannot be improved

The Fisher information is $I(\bar T)=n_{\rm ev}/\bar T^2$ with $n_{\rm ev}=\sum_r\delta_r$,
so the Cramér–Rao bound is

$$\frac{\sigma(\bar T)}{\bar T}\;\ge\;\frac{1}{\sqrt{n_{\rm ev}}} .$$

The exponential has coefficient of variation exactly $1$, *independent of $g$, $N$ and the
barrier height* — fluctuations are not suppressed in any limit. Reaching $5\%$ therefore
requires $\sim400$ escape events, and the only way to do better is to stop measuring
first-passage times and measure the rate instead (umbrella sampling / metadynamics /
forward-flux sampling on the reaction coordinate $\rho$).

**Consistency checks on the exponential assumption**, to be run on any new folder:

$$\frac{\text{median}}{\text{mean}}\ \to\ \ln 2 = 0.6931,
\qquad
\frac{\sigma(T)}{\bar T}\ \to\ 1,
\qquad
P(T<\bar T)\ \to\ 1-e^{-1}=0.632 .$$

The median is estimated non-parametrically by Kaplan–Meier (`km_median`) so that it is
independent of the exponential assumption being tested.

---

## 5. Normalized escape time

### 5.1 Definition

$$\boxed{\;\mathcal{T}\;\equiv\;\frac{\bar T}{\tau_{\rm int}}\;}
\qquad\text{(column \texttt{mean\_tau\_ratio})}$$

Both quantities are in trajectories, so $\mathcal{T}$ is dimensionless. The point is that a
trajectory is an arbitrary unit — it depends on `nsteps`, `step_size` and the integrator —
and here `step_size` $=\sqrt{g}$, so the unit *changes along the scan*. Writing the escape
rate per trajectory in Kramers form,

$$\Gamma \;=\; \nu\; e^{-\beta\Delta F},
\qquad \nu \sim \frac{1}{\tau_{\rm int}}
\quad\Longrightarrow\quad
\mathcal{T}=\frac{1}{\Gamma\,\tau_{\rm int}}\simeq e^{\beta\Delta F},$$

dividing by $\tau_{\rm int}$ removes the algorithmic attempt frequency and leaves the
equilibrium factor. Equivalently: $\mathcal{T}/2$ is the number of *independent*
configurations drawn per escape, i.e. the inverse probability that an equilibrium
configuration in the metastable basin sits in the escape channel.

### 5.2 Uncertainty

Nonparametric bootstrap over replicas, $B = 20{,}000$ resamples. For each resample $b$,
draw $R$ replicas with replacement (carrying their $(T_r,\delta_r)$ pair together),
recompute the MLE, and form

$$\mathcal{T}^{(b)}=\frac{\bar T^{(b)}}{\tau_{\rm int}} .$$

Resamples with zero events are rejected and redrawn. Reported quantities:

$$\sigma_{\mathcal{T}}=\operatorname{sd}_b\big[\mathcal{T}^{(b)}\big],
\qquad
\sigma_{\log}=\operatorname{sd}_b\big[\log\mathcal{T}^{(b)}\big],
\qquad
\text{CI}_{95}=\big[q_{2.5\%},\,q_{97.5\%}\big] .$$

$\tau_{\rm int}$ is held **fixed** across resamples, so $\sigma_{\mathcal{T}}$ measures
first-passage sampling error only. This is a deliberate approximation, justified because
$\sigma(\tau_{\rm int})/\tau_{\rm int}\approx\sqrt{2(2W+1)/N_{\rm tot}}\approx0.5\%$ is an
order of magnitude below the $\approx4.7\%$ error on $\bar T$. Re-check that inequality on
a new dataset before relying on it.

### 5.3 Fit

Per $g$, $\log\mathcal{T}$ is regressed on $1/g$ by weighted least squares
(`wls_fit`, weights $1/\sigma_{\log}^2$):

$$\log\!\left(\frac{\bar T}{\tau_{\rm int}}\right)=A+\frac{B}{g} .$$

The covariance is scaled by $\max(1,\chi^2/\text{dof})$ (PDG convention). Because the
bosonic potential satisfies $V=(N/g)\,v(X)$ with $v$ independent of $g$, the coupling acts
as a temperature, $\beta = N/g$, and

$$B \;=\; N\,\Delta v ,$$

with $\Delta v$ the barrier of the $g$-independent function $v$ — i.e. $B$ is fixed by a
classical saddle-point computation, while $A$ carries the algorithm-dependent prefactor.

---

## 6. Applying this to a new folder

1. **Point the loader at the folder.** Edit `BASES` and `G_VALUES` in
   `analyze_robust_ell1_precision.py`, and edit `run_is_robust_ell1` so its parameter
   checks match the new runs (model, $N$, $\omega$, spin, `nsteps`, `step_size`,
   descent step size, `validation_halvings`).
2. **Run it.**
   ```
   python analyze_robust_ell1_precision.py --out <output_dir> --boot 20000
   ```
   Outputs: `results.csv` (one row per $g$), `replicas.csv` (one row per run, including
   rejected ones), `fit_summary.json`, plots, and `final_report.md`.
3. **Read the diagnostics in `results.csv` before the physics**, in this order:

| column | expectation | meaning if violated |
|---|---|---|
| `number_invalid_excluded` | small vs `number_of_replicas` | descent settings or truncated runs |
| `descent_convergence_fraction` | $=1$ | classifier unreliable; tighten `--escape-descent-steps` |
| `tau_int_diagnostic_window` | $\ll$ segment length | window not resolved; more/longer segments |
| `tau_int_min_pair_count_at_window` | large | $\hat\rho$ at the window is noise-dominated |
| `fraction_T_le_10` | $\approx 0$ | barrier too shallow; $T\gtrsim\tau_{\rm int}$ breaks the Poisson picture |
| `number_censored` / `number_of_replicas` | small | `niters` too short relative to $\bar T$ |
| `median_tau_ratio` / `mean_tau_ratio` | $\approx\ln 2$ | escape not exponential |
| `acceptance_mean` | $\gtrsim 0.6$ | integrator step too large |

4. **Verify the unit is doing its job.** At one fixed $g$, rerun with different `nsteps`
   and `step_size`. Raw $\bar T$ must change; $\mathcal{T}=\bar T/\tau_{\rm int}$ must not.
   If $\mathcal{T}$ drifts, the prefactor $A$ is not algorithm-independent and only $B$
   should be quoted.

---

## 7. Reference values (`metastability_results_robust_ell1_precision5`)

$N=2$, $\omega=1$, bosonic, spin $0$, `nsteps` $=13$, `step_size` $=\sqrt{g}$,
descent step $0.4g$, `grad_tol` $=10^{-3}$, one halved-step validation pass.

| $g$ | $R$ | events | censored | $\tau_{\rm int}$ | $\bar T$ | $\mathcal{T}=\bar T/\tau_{\rm int}$ | rel. err. |
|---|---|---|---|---|---|---|---|
| 0.0052 | 461 | 447 | 14 | 0.9577 | 2312 | 2414 | 4.68% |
| 0.0053 | 455 | 450 | 5 | 0.9801 | 1944 | 1984 | 4.54% |
| 0.0054 | 456 | 449 | 7 | 1.0128 | 1780 | 1757 | 4.92% |
| 0.0055 | 451 | 450 | 1 | 1.0082 | 1455 | 1443 | 4.62% |
| 0.0056 | 542 | 540 | 2 | 1.0246 | 1242 | 1213 | 4.60% |
| 0.0057 | 542 | 539 | 3 | 1.0413 | 1224 | 1175 | 4.61% |
| 0.0058 | 450 | 450 | 0 | 1.0596 | 954 | 900 | 4.66% |
| 0.0059 | 450 | 450 | 0 | 1.0333 | 814 | 788 | 4.67% |

$$A=-1.4488\pm0.3987,\qquad B=0.04803\pm0.00221,\qquad \chi^2/\text{dof}=0.914 .$$

For comparison, fitting the **unnormalized** $\log\bar T$ gives $B=0.04424\pm0.00240$ with
$\chi^2/\text{dof}=1.183$: the normalization shifts $B$ by $8.6\%$ ($1.6\sigma$) and
improves the fit. The straight-line variational bound on the classical barrier along
$X_i=c\,J_i$ gives $\Delta v \le 1/36$ at $c=1/3$, i.e. $B\le N\Delta v = 1/18 = 0.0556$,
consistent with the measurement.
