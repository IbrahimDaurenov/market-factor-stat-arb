# `signals.py`

This module converts PCA residual spreads into trading signals and manages persistent long and short positions.

The final strategy uses the pipeline

```math
e_t
\rightarrow
s_t
\rightarrow
z_t
\rightarrow
p_t
```

where:

- $e_t$ is the PCA residual;
- $s_t$ is the cumulative residual spread;
- $z_t$ is its standardized value;
- $p_t$ is the resulting position state.

---

## Residual Spread

The previous module constructs the residual spread

```math
s_{t,i}
=
\sum_{j=0}^{H-1}
e_{t-j,i}
```

for every stock $i$.

The final strategy uses

```math
H=40
```

days.

A large positive value means that the stock has recently outperformed its PCA-implied component.

A large negative value means that it has underperformed.

The strategy tests whether these relative deviations mean-revert.

---

## Rolling Z-Score

To determine whether the current spread is unusual, it is standardized relative to its own recent history.

For stock $i$, define the historical mean

```math
\mu_{t,i}
=
\frac{1}{W}
\sum_{j=1}^{W}
s_{t-j,i}
```

and historical standard deviation

```math
\sigma_{t,i}
=
\sqrt{
\frac{1}{W-1}
\sum_{j=1}^{W}
(s_{t-j,i}-\mu_{t,i})^2
}
```

Then

```math
z_{t,i}
=
\frac{s_{t,i}-\mu_{t,i}}
{\sigma_{t,i}}
```

The final model uses

```math
W=252
```

trading days.

---

## Why `shift(1)` Is Important

The implementation begins with

```python
past_spreads = spreads.shift(1)
```

This means the rolling mean and standard deviation for day $t$ use only

```math
s_{t-W},\ldots,s_{t-1}
```

and do not include $s_t$ itself.

The chronology is therefore:

```text
past spreads
→ estimate historical mean and volatility
→ observe current spread
→ compute current z-score
```

This avoids using the current observation to define its own historical distribution.

---

## Z-Score Trading Rule

The final strategy uses

```math
z_{\mathrm{entry}}=2.5
```

Long entry:

```math
z_{t,i}<-2.5
```

Short entry:

```math
z_{t,i}>2.5
```

The idea is mean reversion:

- a large negative deviation becomes a long candidate;
- a large positive deviation becomes a short candidate.

This is a trading hypothesis, not a mathematical consequence of PCA.

PCA defines the residual; the backtest tests whether the residual deviations mean-revert.

---

## Position State

For every stock,

```math
p_{t,i}\in\{-1,0,+1\}
```

where:

```text
+1 = long
-1 = short
 0 = flat
```

Positions persist across days rather than being rebuilt from zero every day.

---

## Z-Score Exit Rule

A long position exits when

```math
z_{t,i}\geq0
```

A short position exits when

```math
z_{t,i}\leq0
```

So the basic logic is:

```text
negative extreme → long → return to zero → exit
positive extreme → short → return to zero → exit
```

---

## Maximum Holding Period

A position is also closed if it remains open for too long.

Let

```math
h_{t,i}
```

be the number of days the position has been held.

The position exits when

```math
h_{t,i}\geq H_{\max}
```

The final strategy uses

```math
H_{\max}=40
```

days.

This prevents a failed mean-reversion trade from remaining open indefinitely.

---

## Maximum Number of Positions

The strategy limits the number of simultaneous active positions.

Let

```math
A_t
=
\sum_i
\mathbf{1}(p_{t,i}\neq0)
```

Then

```math
A_t\leq A_{\max}
```

The final model uses

```math
A_{\max}=10
```

---

## Ranking Simultaneous Signals

More than 10 stocks can satisfy the entry rule on the same day.

For the z-score strategy, signal strength is

```math
|z_{t,i}|
```

Candidates are sorted from largest to smallest absolute z-score.

The most extreme residual deviations are therefore selected first.

---

## Exit Before Entry

`update_positions()` first processes existing positions.

Only after exits are handled does it search for new entries.

The sequence is:

```text
existing positions
→ exits
→ available slots
→ new entries
```

Closing a position can therefore free capacity for a new trade.

---

## No Same-Day Re-entry

If a stock exits on day $t$, it is stored in `exited_today`.

That stock cannot immediately reopen on the same day.

This avoids:

```text
exit → immediate re-entry
```

from the same observation.

---

## Empirical Quantile Alternative

A second signal definition was tested without using z-scores.

Instead of standardizing the spread, the historical empirical distribution is used directly.

For each stock, the model estimates:

```math
q_{\mathrm{low},t},
\qquad
q_{50,t},
\qquad
q_{\mathrm{high},t}
```

from the previous $W$ observations.

The implementation uses

```math
q_{\mathrm{low}}=0.025
```

and

```math
q_{\mathrm{high}}=0.975
```

so the entry thresholds correspond to the historical 2.5% and 97.5% tails.

---

## Quantile Entry Rule

Long entry:

```math
s_{t,i}<q_{\mathrm{low},t,i}
```

Short entry:

```math
s_{t,i}>q_{\mathrm{high},t,i}
```

---

## Quantile Exit Rule

Long positions exit when

```math
s_{t,i}\geq q_{50,t,i}
```

Short positions exit when

```math
s_{t,i}\leq q_{50,t,i}
```

The historical median acts as the center of the distribution.

---

## Quantile Signal Strength

For simultaneous candidates, signal strength is

```math
\frac{|s_t-q_{50}|}
{q_{\mathrm{high}}-q_{\mathrm{low}}}
```

The denominator normalizes the deviation by the historical width of the distribution.

This allows signals from different stocks to be compared on approximately the same scale.

---

## Quantiles vs Z-Scores

The empirical-quantile version was tested as a robustness experiment.

In the final backtest it produced:

- more active positions;
- higher turnover;
- higher transaction costs;
- lower Sharpe ratio.

The final model therefore uses the z-score signal.

---

## AR(1) Diagnostic

An AR(1) model was also tested as an optional mean-reversion filter.

The implementation does not fit AR(1) directly to the daily residual $e_t$.

It first creates the cumulative residual level

```math
S_t
=
\sum_{\tau\leq t}e_\tau
```

and then models

```math
S_t
=
\alpha
+
\phi S_{t-1}
+
\epsilon_t
```

---

## Estimating Phi

For the regression

```math
Y=\alpha+\phi X+\epsilon
```

the OLS slope is

```math
\phi
=
\frac{\mathrm{Cov}(X,Y)}
{\mathrm{Var}(X)}
```

Here,

```math
X=S_{t-1}
```

and

```math
Y=S_t
```

so the rolling estimator is

```math
\hat{\phi}
=
\frac{
\mathrm{Cov}(S_{t-1},S_t)
}{
\mathrm{Var}(S_{t-1})
}
```

The intercept is

```math
\hat{\alpha}
=
\bar{S}_t
-
\hat{\phi}\bar{S}_{t-1}
```

---

## AR(1) Mean Reversion

For

```math
|\phi|<1
```

the AR(1) process is stationary.

Ignoring the intercept for intuition,

```math
E[S_{t+h}\mid S_t]
\approx
\phi^hS_t
```

When

```math
0<\phi<1
```

the effect of a deviation decays over time.

---

## Half-Life

Half-life is the number of periods required for the expected deviation to fall to one half of its original size.

We require

```math
\phi^h=\frac{1}{2}
```

Taking logarithms,

```math
h\log(\phi)=\log(1/2)
```

so

```math
h_{1/2}
=
\frac{\log(1/2)}
{\log(\phi)}
```

This is meaningful in the implementation only when

```math
0<\phi<1
```

---

## Why the AR(1) Filter Was Not Used

In the data, estimated $\phi$ was usually close to one.

There is also a structural reason for this.

Since

```math
S_t=S_{t-1}+e_t
```

a cumulative sum of approximately zero-mean residuals naturally behaves similarly to a random walk.

A random walk corresponds approximately to

```math
\phi=1
```

Therefore a value of $\phi$ close to one for the cumulative residual level is not strong evidence for short-term mean reversion.

The AR(1) filter also reduced backtest performance.

The final specification therefore uses

```python
use_ar1_filter = False
```

The AR(1) model remains in the project as a diagnostic and a failed experiment.

---

## Optional AR Filter in Position Selection

If `phi_today` is supplied, a new entry is accepted only when

```math
\phi_{\min}<\phi_t<\phi_{\max}
```

If no AR parameters are supplied,

```python
phi_today = None
```

the filter is skipped completely.

This is the configuration used by the final strategy.

---

## Final Signal Specification

The final strategy uses:

```math
H=40
```

for residual accumulation,

```math
W=252
```

for z-score estimation,

```math
z_{\mathrm{entry}}=2.5
```

for entries,

```math
H_{\max}=40
```

for maximum holding time, and

```math
A_{\max}=10
```

for the maximum number of simultaneous positions.

The final pipeline is

```math
e_t
\rightarrow
s_t
\rightarrow
z_t
\rightarrow
p_t
```

The mapping from $z_t$ to $p_t$ is determined by the entry and exit rules.

AR(1) and empirical quantiles remain alternative experiments rather than components of the final trading model.
