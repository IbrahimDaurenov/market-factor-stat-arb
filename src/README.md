# Source Code: Mathematical Reference

This file explains the mathematics behind the modules in `src/`.

The main pipeline is

$$
\text{prices}
\rightarrow
\text{returns}
\rightarrow
\text{PCA}
\rightarrow
\text{residuals}
\rightarrow
\text{signals}
\rightarrow
\text{portfolio}
\rightarrow
\text{backtest}.
$$

---

# `data.py`

## Prices and Returns

For stock \(i\), let \(P_{t,i}\) be its closing price on day \(t\).

The simple daily return is

$$
r_{t,i}
=
\frac{P_{t,i}}{P_{t-1,i}}-1.
$$

For \(N\) stocks, one trading day is represented by

$$
x_t
=
\begin{bmatrix}
r_{t,1}\\
r_{t,2}\\
\vdots\\
r_{t,N}
\end{bmatrix}
\in\mathbb{R}^N.
$$

Stacking \(T\) observations gives the return matrix

$$
X
=
\begin{bmatrix}
x_1^\top\\
x_2^\top\\
\vdots\\
x_T^\top
\end{bmatrix}
\in\mathbb{R}^{T\times N}.
$$

In this project:

- rows = trading days,
- columns = stocks.

The final model trades only the stock universe. A broad-market series may still be loaded for diagnostics, but the final portfolio uses PCA factor neutralization instead of a separate market hedge.

---

# `pca.py`

## 1. Centering

PCA is applied to centered returns.

The sample mean is

$$
\mu
=
\frac{1}{T}
\sum_{t=1}^{T}x_t.
$$

Each observation becomes

$$
z_t=x_t-\mu.
$$

The centered matrix is

$$
X_c
=
X-\mathbf{1}\mu^\top.
$$

Therefore each column of \(X_c\) has mean approximately zero.

---

## 2. Covariance Matrix

The sample covariance matrix is

$$
\Sigma
=
\frac{1}{T-1}X_c^\top X_c.
$$

Its entries are

$$
\Sigma_{ij}
=
\frac{1}{T-1}
\sum_{t=1}^{T}
z_{t,i}z_{t,j}.
$$

Thus

$$
\Sigma_{ii}=\operatorname{Var}(X_i)
$$

and

$$
\Sigma_{ij}=\operatorname{Cov}(X_i,X_j).
$$

The matrix is symmetric:

$$
\Sigma^\top=\Sigma.
$$

It is also positive semidefinite because for every vector \(a\),

$$
a^\top\Sigma a
=
\frac{1}{T-1}
a^\top X_c^\top X_c a
=
\frac{1}{T-1}
\|X_ca\|^2
\geq0.
$$

Therefore all eigenvalues are non-negative.

---

## 3. Eigendecomposition

Since \(\Sigma\) is real and symmetric,

$$
\Sigma
=
Q\Lambda Q^\top,
$$

where

$$
Q^\top Q=I.
$$

The columns of \(Q\) are eigenvectors:

$$
Q=
[v_1,\ldots,v_N].
$$

Each satisfies

$$
\Sigma v_i
=
\lambda_i v_i.
$$

The eigenvalues are sorted in descending order:

$$
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_N.
$$

---

## 4. Why Eigenvectors Give PCA Directions

Take any unit vector \(v\):

$$
\|v\|=1.
$$

The projection of a centered return vector \(z\) onto \(v\) is

$$
c=v^\top z.
$$

Because \(E[z]=0\),

$$
\operatorname{Var}(c)
=
E[(v^\top z)^2].
$$

Now

$$
(v^\top z)^2
=
v^\top zz^\top v.
$$

Therefore

$$
\operatorname{Var}(c)
=
v^\top E[zz^\top]v
=
v^\top\Sigma v.
$$

PCA chooses the unit direction that maximizes this quantity:

$$
\max_{\|v\|=1}v^\top\Sigma v.
$$

The solution is the eigenvector \(v_1\) corresponding to the largest eigenvalue.

For an eigenvector,

$$
\operatorname{Var}(v_i^\top z)
=
v_i^\top\Sigma v_i
=
\lambda_i.
$$

Therefore the eigenvalue is exactly the variance explained in that principal direction.

---

## 5. Explained Variance

Total variance is

$$
\operatorname{tr}(\Sigma)
=
\sum_{j=1}^{N}\lambda_j.
$$

The explained-variance ratio of PC \(i\) is

$$
EVR_i
=
\frac{\lambda_i}
{\sum_{j=1}^{N}\lambda_j}.
$$

For the first \(k\) components,

$$
EVR_{1:k}
=
\frac{\lambda_1+\cdots+\lambda_k}
{\sum_{j=1}^{N}\lambda_j}.
$$

The final model uses

$$
k=6.
$$

---

## 6. PCA Scores

Let

$$
Q_k=[v_1,\ldots,v_k].
$$

For one centered observation \(z_t\), the PCA scores are

$$
c_t
=
Q_k^\top z_t.
$$

The \(i\)-th score is

$$
c_{t,i}
=
v_i^\top z_t.
$$

A loading is a coordinate of an eigenvector.

A score tells how strongly that eigenvector is present on a specific day.

---

# `residuals.py`

## PCA Reconstruction

The projection matrix onto the first \(k\) principal components is

$$
P_k
=
Q_kQ_k^\top.
$$

Because the columns of \(Q_k\) are orthonormal,

$$
P_k^\top=P_k
$$

and

$$
P_k^2=P_k.
$$

The common-factor part of a return vector is

$$
\hat z_t
=
P_kz_t
=
Q_kQ_k^\top z_t.
$$

The PCA residual is

$$
e_t
=
z_t-\hat z_t.
$$

Hence

$$
e_t
=
(I-Q_kQ_k^\top)z_t.
$$

The residual is orthogonal to the retained PCA subspace:

$$
Q_k^\top e_t=0.
$$

So we obtain the decomposition

$$
z_t
=
Q_kQ_k^\top z_t
+
(I-Q_kQ_k^\top)z_t.
$$

Interpretation:

$$
\text{return}
=
\text{common PCA component}
+
\text{residual}.
$$

---

## Rolling PCA

A PCA model estimated using the entire dataset would use future information.

Therefore for day \(t\), only the previous \(L\) observations are used:

$$
X_t^{train}
=
X_{t-L:t-1}.
$$

For each day:

$$
X_t^{train}
\rightarrow
\mu_t
\rightarrow
\Sigma_t
\rightarrow
Q_{k,t}.
$$

Then today's centered return is

$$
z_t=x_t-\mu_t
$$

and the residual is

$$
e_t
=
(I-Q_{k,t}Q_{k,t}^\top)z_t.
$$

The final project uses

$$
L=252.
$$

---

## Residual Spread

A single daily residual can contain noise.

The strategy accumulates residuals over \(H\) days:

$$
s_t
=
\sum_{j=0}^{H-1}e_{t-j}.
$$

For stock \(i\),

$$
s_{t,i}
=
\sum_{j=0}^{H-1}e_{t-j,i}.
$$

The final specification uses

$$
H=40.
$$

This parameter was found to be sensitive, so the result should be interpreted carefully.

---

# `signals.py`

## Rolling Z-Score

For every stock, estimate the mean and standard deviation of the residual spread using only past observations.

Let the signal-estimation window be \(W\).

Then

$$
\mu_{t,i}
=
\frac{1}{W}
\sum_{j=1}^{W}s_{t-j,i}
$$

and

$$
\sigma_{t,i}
=
\sqrt{
\frac{1}{W-1}
\sum_{j=1}^{W}
(s_{t-j,i}-\mu_{t,i})^2
}.
$$

The z-score is

$$
z_{t,i}
=
\frac{s_{t,i}-\mu_{t,i}}
{\sigma_{t,i}}.
$$

The final model uses

$$
W=252.
$$

Entry rule:

$$
z_{t,i}<-2.5
\Rightarrow
\text{long}
$$

and

$$
z_{t,i}>2.5
\Rightarrow
\text{short}.
$$

The economic assumption is mean reversion:

an unusually negative relative deviation may move upward later, while an unusually positive deviation may move downward.

---

## Position State

For every stock,

$$
p_{t,i}\in\{-1,0,+1\}.
$$

Here

$$
p_{t,i}=+1
$$

means long,

$$
p_{t,i}=-1
$$

means short,

and

$$
p_{t,i}=0
$$

means no position.

Positions remain active until an exit condition is reached or the maximum holding period expires.

The final model uses

$$
H_{max}=40
$$

trading days.

---

## Empirical Quantiles

A non-parametric alternative was also tested.

Instead of assuming that a fixed z-score identifies an extreme observation, calculate rolling empirical thresholds:

$$
q_{low,t}
=
Q_{\alpha}
(s_{t-W},\ldots,s_{t-1})
$$

and

$$
q_{high,t}
=
Q_{1-\alpha}
(s_{t-W},\ldots,s_{t-1}).
$$

Then

$$
s_t<q_{low,t}
$$

is a long candidate and

$$
s_t>q_{high,t}
$$

is a short candidate.

This approach was tested but produced more turnover and a lower Sharpe ratio than the final z-score signal.

---

## AR(1) Filter

An AR(1) model was tested as an additional mean-reversion diagnostic.

For a residual series,

$$
e_t
=
\alpha+\phi e_{t-1}+\epsilon_t.
$$

If

$$
|\phi|<1,
$$

a shock decays over time.

Ignoring the intercept for intuition,

$$
E[e_{t+h}\mid e_t]
\approx
\phi^h e_t.
$$

For

$$
0<\phi<1,
$$

the approximate half-life is

$$
h_{1/2}
=
\frac{\log(1/2)}
{\log(\phi)}.
$$

If \(\phi\) is close to one, mean reversion is slow.

In this project, estimated \(\phi\) values were usually close to one, and the AR(1) filter did not improve the final strategy.

Therefore

```python
use_ar1_filter = False
```

in the final model.

---

# `portfolio.py`

## Raw Weights

If the position state is \(p_i\) and each position receives size \(a\),

$$
w_i^{raw}
=
a p_i.
$$

For example, with

$$
a=0.05,
$$

a long position starts with weight

$$
+0.05
$$

and a short position with

$$
-0.05.
$$

The final strategy allows up to 10 active positions.

---

## PCA Factor Exposure

Let the retained factor matrix be

$$
Q_k
\in
\mathbb{R}^{N\times k}.
$$

Portfolio exposure to the PCA factors is

$$
f
=
Q_k^\top w.
$$

If

$$
Q_k^\top w=0,
$$

the portfolio is orthogonal to all retained PCA directions.

---

## Factor Neutralization

The orthogonal projection onto the factor space is

$$
P_k
=
Q_kQ_k^\top.
$$

The projection onto its orthogonal complement is

$$
I-P_k.
$$

Therefore neutralized weights are

$$
w
=
(I-Q_kQ_k^\top)w^{raw}.
$$

Check:

$$
Q_k^\top w
=
Q_k^\top
(I-Q_kQ_k^\top)
w^{raw}.
$$

Expanding,

$$
Q_k^\top w
=
Q_k^\top w^{raw}
-
Q_k^\top Q_kQ_k^\top w^{raw}.
$$

Since

$$
Q_k^\top Q_k=I,
$$

we obtain

$$
Q_k^\top w
=
Q_k^\top w^{raw}
-
Q_k^\top w^{raw}
=
0.
$$

Thus the portfolio has zero exposure to the retained PCA factors before scaling.

This is the final project's factor hedge.

No separate SPY hedge is used in the final specification.

---

## Gross Exposure

Gross exposure is

$$
G_t
=
\sum_i|w_{t,i}|.
$$

The final strategy limits

$$
G_t\leq0.50.
$$

If neutralized weights exceed this value, they are scaled:

$$
w^{scaled}
=
w
\frac{G_{max}}{\sum_i|w_i|}.
$$

---

## Net Exposure

Net exposure is

$$
N_t
=
\sum_i w_{t,i}.
$$

A portfolio with

$$
N_t=0
$$

is dollar-neutral.

PCA factor neutrality and dollar neutrality are different concepts.

The project explicitly controls PCA factor exposure; exact dollar neutrality is not guaranteed by the PCA projection alone.

---

## Turnover

Turnover is defined as total traded notional relative to capital:

$$
\tau_t
=
\sum_i
|w_{t,i}-w_{t-1,i}|.
$$

Example:

changing from

$$
(0.5,-0.5,0,0)
$$

to

$$
(0,0,0.5,-0.5)
$$

produces

$$
\tau=2.
$$

This means total trades equal 200% of portfolio capital.

---

# `backtest.py`

## Chronological Backtest

The backtest follows the timeline

$$
X_{t-L:t-1}
\rightarrow
Q_{k,t}
\rightarrow
x_t
\rightarrow
e_t
\rightarrow
s_t
\rightarrow
z_t
\rightarrow
w_t
\rightarrow
r_{t+1}.
$$

The signal formed on day \(t\) is applied to the return on day \(t+1\).

Therefore

$$
R_{t+1}^{gross}
=
w_t^\top r_{t+1}.
$$

This shift prevents the strategy from earning a return using information that was unavailable when the trade was formed.

---

## PC1 Explained-Variance Share

At every date,

$$
m_t
=
\frac{\lambda_{1,t}}
{\sum_j\lambda_{j,t}}.
$$

This measures how much of total cross-sectional variance is explained by the first principal component.

A high value means that stock returns are moving more strongly along one common market direction.

---

## PC1 Regime Filter

The current PC1 share is compared with a rolling threshold based only on earlier data.

Let

$$
q_t
=
\text{median}
(
m_{t-252},
\ldots,
m_{t-1}
).
$$

New positions are allowed only if

$$
m_t>q_t.
$$

Thus

$$
\text{high-PC1 regime}
\Rightarrow
\text{new entries allowed}
$$

and

$$
\text{low-PC1 regime}
\Rightarrow
\text{no new entries}.
$$

Existing positions are not automatically closed when the regime changes.

They continue until their normal exit rule or maximum holding period.

---

## Rebalancing

The portfolio is re-neutralized when positions change or when the scheduled PCA rebalance interval is reached.

The final specification uses

$$
20
$$

trading days as the scheduled rebalance interval.

This reduces unnecessary turnover compared with rebuilding the factor-neutral portfolio every day.

---

## Transaction Costs

The cost rate corresponding to \(b\) basis points is

$$
c
=
\frac{b}{10000}.
$$

For

$$
b=10,
$$

we have

$$
c=0.001.
$$

Transaction cost on day \(t\) is

$$
C_t
=
c\tau_t.
$$

Therefore

$$
R_{t+1}^{net}
=
R_{t+1}^{gross}
-
C_t.
$$

---

# `metrics.py`

## Equity Curve

Starting from capital \(V_0\),

$$
V_{t+1}
=
V_t(1+R_{t+1}).
$$

Therefore

$$
V_T
=
V_0
\prod_{t=1}^{T}
(1+R_t).
$$

Cumulative return is

$$
R_{cum}
=
\frac{V_T}{V_0}-1.
$$

---

## Annualized Volatility

Let

$$
\sigma_d
=
\operatorname{Std}(R_t)
$$

be daily return volatility.

Using approximately 252 trading days per year,

$$
\sigma_{annual}
=
\sqrt{252}\sigma_d.
$$

---

## Sharpe Ratio

With risk-free rate approximated as zero,

$$
Sharpe
=
\frac{E[R_t]}
{\operatorname{Std}(R_t)}
\sqrt{252}.
$$

This measures average return relative to volatility.

---

## Drawdown

Define the running portfolio peak

$$
M_t
=
\max_{s\leq t}V_s.
$$

Drawdown is

$$
D_t
=
\frac{V_t-M_t}{M_t}.
$$

Maximum drawdown is

$$
MDD
=
\min_t D_t.
$$

---

# Final Model

The final specification is

$$
k=6,
\qquad
H=40,
\qquad
z_{entry}=2.5.
$$

Position rules:

$$
\text{max positions}=10,
\qquad
\text{position size}=5\%,
$$

$$
\text{max holding}=40.
$$

Risk and execution:

$$
G_{max}=50\%,
$$

$$
\text{transaction cost}=10\text{ bps},
$$

$$
\text{PCA rebalance}=20\text{ days}.
$$

Regime rule:

$$
m_t>
\text{median of previous 252 PC1 shares}.
$$

The final strategy therefore combines

$$
\boxed{
\text{PCA decomposition}
+
\text{residual mean reversion}
+
\text{factor neutralization}
+
\text{regime filtering}
}
$$

inside a chronological backtest.