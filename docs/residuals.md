# `residuals.py`

This module removes the common PCA factor component from stock returns.

The main idea is

```math
\text{return}
=
\text{PCA factor component}
+
\text{residual}
```

---

## Centered Return

Let

```math
x_t\in\mathbb{R}^N
```

be the vector of stock returns on day $t$.

PCA was estimated using historical data with mean vector

```math
\mu_t
```

The centered return is

```math
z_t=x_t-\mu_t
```

---

## Retained PCA Subspace

Let

```math
Q_k=
[v_1,\ldots,v_k]
```

contain the first $k$ PCA eigenvectors.

Because the eigenvectors are orthonormal,

```math
Q_k^\top Q_k=I
```

The matrix

```math
P_k=Q_kQ_k^\top
```

is the orthogonal projection matrix onto the retained PCA factor space.

---

## Factor Component

The PCA coordinates of $z_t$ are

```math
c_t=Q_k^\top z_t
```

Reconstructing from these coordinates gives

```math
\hat z_t=Q_kc_t
```

Substituting the score formula,

```math
\hat z_t
=
Q_kQ_k^\top z_t
```

This is the part of the return explained by the retained PCA factors.

In code:

```python
factor_component = (
    Q_k @ (Q_k.T @ x_centered)
)
```

---

## PCA Residual

The residual is the part not explained by the retained PCA factors:

```math
e_t
=
z_t-\hat z_t
```

Therefore,

```math
e_t
=
z_t-Q_kQ_k^\top z_t
```

or

```math
\boxed{
e_t
=
(I-Q_kQ_k^\top)z_t
}
```

This is exactly what `compute_residual()` calculates.

---

## Orthogonality of the Residual

The residual must be orthogonal to the retained PCA factors.

Start with

```math
e_t
=
(I-Q_kQ_k^\top)z_t
```

Multiply by $Q_k^\top$:

```math
Q_k^\top e_t
=
Q_k^\top
(I-Q_kQ_k^\top)
z_t
```

Expand:

```math
Q_k^\top e_t
=
Q_k^\top z_t
-
Q_k^\top Q_kQ_k^\top z_t
```

Since

```math
Q_k^\top Q_k=I
```

we obtain

```math
Q_k^\top e_t
=
Q_k^\top z_t
-
Q_k^\top z_t
=
0
```

Therefore,

```math
\boxed{
Q_k^\top e_t=0
}
```

So the residual contains no component in the retained PCA directions.

---

## Geometric Interpretation

The centered return vector can be decomposed as

```math
z_t
=
Q_kQ_k^\top z_t
+
(I-Q_kQ_k^\top)z_t
```

or

```math
z_t
=
\hat z_t+e_t
```

The two components are orthogonal:

```math
\hat z_t^\top e_t=0
```

So geometrically:

- $\hat z_t$ lies inside the PCA factor subspace;
- $e_t$ lies in the orthogonal residual subspace.

---

## Why Rolling PCA Is Necessary

If PCA were fitted using the entire dataset, the model would use future returns to estimate today's eigenvectors.

That would introduce look-ahead bias.

Instead, for each day $t$, PCA is fitted only on the previous $L$ observations:

```math
X_t^{train}
=
\{x_{t-L},\ldots,x_{t-1}\}
```

The final project uses

```math
L=252
```

trading days.

The sequence is

```math
X_{t-252:t-1}
\rightarrow
\mu_t
\rightarrow
\Sigma_t
\rightarrow
Q_{k,t}
```

Then today's return $x_t$ is observed and residualized:

```math
z_t=x_t-\mu_t
```

```math
e_t
=
(I-Q_{k,t}Q_{k,t}^\top)z_t
```

No future return is used.

---

## Out-of-Sample Residual

This distinction is important.

The PCA basis

```math
Q_{k,t}
```

is estimated using data ending at day $t-1$.

Then it is applied to day $t$.

So $e_t$ is an out-of-sample residual relative to the historical PCA model available at that time.

---

## First 252 Observations

The model needs a full PCA estimation window before producing residuals.

Therefore the first 252 rows are unavailable:

```math
e_t=\mathrm{NaN}
\qquad
t<252
```

After that point, one residual vector is generated per trading day.

---

## Residual Spread

A one-day residual can be noisy.

Instead of trading directly on $e_t$, the strategy accumulates residuals over the previous $H$ days:

```math
s_t
=
\sum_{j=0}^{H-1}e_{t-j}
```

For stock $i$,

```math
s_{t,i}
=
e_{t,i}
+
e_{t-1,i}
+
\cdots
+
e_{t-H+1,i}
```

This is what `rolling_residual_spread()` computes.

---

## Interpretation of the Spread

Suppose a stock repeatedly underperforms relative to the PCA factor model.

Then its residuals may be negative across several days.

Accumulating them produces a strongly negative spread:

```math
s_{t,i}\ll0
```

Similarly, persistent relative outperformance may produce

```math
s_{t,i}\gg0
```

The strategy later tests whether unusually large residual spreads tend to mean-revert.

---

## Residual Horizon

`rolling_residual_spread()` is written as a general function:

```python
rolling_residual_spread(
    residuals,
    window=20
)
```

but the final strategy calls it with

```math
H=40
```

The final signal therefore uses

```math
s_t
=
\sum_{j=0}^{39}e_{t-j}
```

The backtest was sensitive to this parameter, so $H=40$ should not be interpreted as a universally optimal horizon.

---

## Connection to `portfolio.py`

The same projection appears again when constructing portfolio weights.

For returns:

```math
e_t
=
(I-Q_kQ_k^\top)z_t
```

For portfolio weights:

```math
w_t
=
(I-Q_kQ_k^\top)w_t^{raw}
```

So both the signal and the portfolio are built in the same PCA residual subspace.