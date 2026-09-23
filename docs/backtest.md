# `backtest.py`

This module combines all parts of the strategy into one chronological walk-forward backtest.

The main pipeline is

```math
\text{returns}
\rightarrow
\text{PCA residuals}
\rightarrow
\text{residual spread}
\rightarrow
\text{signal}
\rightarrow
\text{positions}
\rightarrow
\text{factor-neutral weights}
\rightarrow
\text{next-day return}
```

The important rule is that the backtest must never use future information.

---

## Rolling PCA Residuals

For each day $t$, PCA is fitted only on the previous `pca_window` observations:

```math
X_{t-L:t-1}
```

where

```math
L=252
```

in the final model.

From this historical window we estimate

```math
\mu_t,\quad \Sigma_t,\quad Q_{k,t}
```

and compute today's residual

```math
e_t
=
(I-Q_{k,t}Q_{k,t}^{\top})
(x_t-\mu_t)
```

The final strategy keeps

```math
k=6
```

principal components.

---

## Residual Spread

The daily residuals are accumulated over a rolling horizon:

```math
s_t
=
\sum_{j=0}^{H-1} e_{t-j}
```

The final model uses

```math
H=40
```

days.

This produces the residual spread used by the trading signal.

---

## Signal Statistics

The final strategy uses a rolling z-score.

For each stock:

```math
z_t
=
\frac{s_t-\mu_t^{(s)}}
{\sigma_t^{(s)}}
```

where the historical mean and standard deviation use only previous spread observations.

The code uses

```python
past_spreads = spreads.shift(1)
```

so the statistics for day $t$ are based on

```math
s_{t-W},\ldots,s_{t-1}
```

rather than including $s_t$ itself.

The final window is

```math
W=252
```

days.

---

## Entry Rule

The z-score entry threshold is

```math
z_{\text{entry}}=2.5
```

A stock becomes a long candidate if

```math
z_{t,i}<-2.5
```

and a short candidate if

```math
z_{t,i}>2.5
```

The signal assumes that extreme PCA residual deviations may mean-revert.

---

## Alternative Signal

The backtest also supports empirical quantile thresholds.

For each stock, historical lower, median, and upper thresholds are estimated from past data:

```math
q_{low,t},\quad q_{50,t},\quad q_{high,t}
```

The default tail probabilities are

```math
q_{low}=0.025
```

and

```math
q_{high}=0.975
```

This version was tested as an alternative but is not used by the final strategy.

---

## AR(1) Diagnostic

The backtest can optionally use the AR(1) parameters from `signals.py`.

The model is

```math
S_t
=
\alpha+\phi S_{t-1}+\epsilon_t
```

where $S_t$ is the cumulative residual level.

If enabled, new entries are accepted only when

```math
\phi_{\min}<\phi_t<\phi_{\max}
```

The final model uses

```python
use_ar1_filter = False
```

because the AR(1) filter did not improve performance.

---

# PC1 Regime Filter

The strategy also measures the strength of the first PCA factor.

For each day $t$, PCA is fitted using only the previous 252 days.

The PC1 explained-variance share is

```math
m_t
=
\frac{\lambda_{1,t}}
{\sum_j \lambda_{j,t}}
```

A large $m_t$ means that a large fraction of total stock-return variance is explained by one common PCA direction.

---

## Past-Only Regime Threshold

The current PC1 share is compared with its historical rolling median.

Define

q_t = \text{median}(m_{t-252}, \dots, m_{t-1})

The code shifts the PC1 series before calculating the threshold, so today's value is not used inside its own benchmark.

The regime is high when

```math
m_t>q_t
```

and low otherwise.

---

## Effect of the Regime Filter

In a high-PC1 regime:

```math
\text{new entries are allowed}
```

In a low-PC1 regime:

```math
\text{new entries are blocked}
```

Existing positions are not automatically closed.

They are still processed using the usual exit rules and maximum holding period. :contentReference[oaicite:2]{index=2}

This is implemented by setting

```python
allowed_max_positions = 0
```

during a low-PC1 regime.

---

# Position Management

The backtest maintains persistent position states

```math
p_{t,i}\in\{-1,0,+1\}
```

for every stock.

The final model uses:

```math
\text{max positions}=10
```

```math
\text{max holding}=40
```

```math
\text{position size}=5\%
```

Existing positions are updated first.

Then free portfolio slots can be used for new entries.

---

# Rolling PCA Model for Portfolio Construction

After the position state for day $t$ is determined, the backtest fits another PCA model using

```math
X_{t-252:t-1}
```

This produces the current factor basis

```math
Q_{k,t}
```

used for portfolio neutralization. :contentReference[oaicite:3]{index=3}

---

# Raw Portfolio Weights

Position states are converted into raw weights:

```math
w_{t,i}^{raw}
=
a p_{t,i}
```

with

```math
a=0.05
```

in the final model.

So:

```math
p_{t,i}=+1
\Rightarrow
w_{t,i}^{raw}=+0.05
```

and

```math
p_{t,i}=-1
\Rightarrow
w_{t,i}^{raw}=-0.05
```

---

# PCA Factor Neutralization

Raw weights may still contain exposure to the retained PCA directions.

The neutralized portfolio is

```math
w_t
=
(I-Q_{k,t}Q_{k,t}^{\top})
w_t^{raw}
```

Therefore,

```math
Q_{k,t}^{\top}w_t=0
```

immediately after neutralization.

This is the factor hedge used by the final strategy.

---

# Rebalancing

The factor-neutral portfolio is not rebuilt every day.

A rebalance occurs when either:

1. the position state changes, or
2. the scheduled rebalance interval is reached.

The final model uses

```math
\text{rebalance interval}=20
```

trading days.

This reduces unnecessary turnover.

---

## Factor Exposure Between Rebalances

If the portfolio is not rebalanced on day $t$, the previous weights are carried forward.

However, the PCA basis

```math
Q_{k,t}
```

can change over time.

Therefore the factor exposure

```math
Q_{k,t}^{\top}w_t
```

may drift slightly between rebalances.

The backtest records

```python
max_factor_exposure
```

as a diagnostic.

---

# Gross Exposure Limit

Gross portfolio exposure is

```math
G_t
=
\sum_i |w_{t,i}|
```

The final strategy limits gross exposure to

```math
G_{\max}=0.50
```

If neutralized weights exceed this level, all weights are scaled proportionally.

---

# Turnover

Turnover is

```math
\tau_t
=
\sum_i
|w_{t,i}-w_{t-1,i}|
```

The backtest records both:

```text
raw_turnover
```

and

```text
turnover
```

where `turnover` is based on the final PCA-neutralized portfolio.

The difference

```math
\text{extra PCA turnover}
=
\text{turnover}
-
\text{raw turnover}
```

measures additional trading caused by factor neutralization.

---

# Transaction Costs

If trading costs are $b$ basis points per unit of turnover,

```math
c=\frac{b}{10000}
```

For the final strategy,

```math
b=10
```

so

```math
c=0.001
```

Transaction cost on day $t$ is

```math
C_t
=
c\tau_t
```

---

# Backtest Timing

This is one of the most important parts of the implementation.

The strategy forms weights using information available on day $t$.

Those weights are applied to returns on day $t+1$.

The chronology is

```math
\text{past data}
\rightarrow
\text{signal at }t
\rightarrow
w_t
\rightarrow
r_{t+1}
```

Gross portfolio return is

```math
R_{t+1}^{gross}
=
w_t^{\top}r_{t+1}
```

and net return is

```math
R_{t+1}^{net}
=
R_{t+1}^{gross}
-
C_t
```

This avoids using day $t+1$ returns when constructing day $t$ positions. :contentReference[oaicite:4]{index=4}

---

# Saved Backtest History

For every trading day, the backtest stores:

```text
gross return
net return
transaction cost
turnover
raw turnover
extra PCA turnover
net exposure
gross exposure
active positions
factor exposure
rebalance indicator
PC1 share
PC1 threshold
regime state
```

It also stores the complete position and weight history. :contentReference[oaicite:5]{index=5}

---

# Output

`run_backtest()` returns a dictionary containing:

```text
residuals
spreads
zscores
quantile thresholds
AR(1) parameters
PC1 regime series
positions
weights
performance
```

The main performance DataFrame contains the realized next-day strategy returns and portfolio diagnostics. :contentReference[oaicite:6]{index=6}

---

# Final Strategy Configuration

The final specification is:

```text
PCA window            252
Principal components    6
Residual horizon        40
Z-score window         252
Entry z-score          2.5
AR(1) filter           OFF
PC1 regime filter       ON
Regime window          252
Regime threshold       median
Max holding             40
Position size           5%
Max positions           10
Max gross exposure      50%
Transaction cost        10 bps
Rebalance interval      20 days
```

The full chronological pipeline is

```math
X_{t-252:t-1}
\rightarrow
Q_{k,t}
\rightarrow
e_t
\rightarrow
s_t
\rightarrow
z_t
\rightarrow
\text{regime filter}
\rightarrow
p_t
\rightarrow
w_t
\rightarrow
r_{t+1}
```

This is the core execution logic of the project.
