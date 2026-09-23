# `portfolio.py`

This module converts trading signals into portfolio weights and removes exposure to the retained PCA factors.

The main pipeline is

```math
p_t
\rightarrow
w_t^{raw}
\rightarrow
w_t^{neutral}
```

---

## Position States

For each stock,

```math
p_i\in\{-1,0,+1\}
```

where

```math
p_i=+1
```

means long,

```math
p_i=-1
```

means short, and

```math
p_i=0
```

means no position.

---

## Raw Portfolio Weights

Let the fixed position size be $a$.

Then

```math
w_i^{raw}=a p_i
```

For the final strategy,

```math
a=0.05
```

so

```math
p_i=+1
\Rightarrow
w_i^{raw}=+0.05
```

and

```math
p_i=-1
\Rightarrow
w_i^{raw}=-0.05
```

The raw portfolio therefore contains the direct long/short decisions produced by the signal.

---

## Why Raw Weights Still Have Factor Exposure

The PCA factor matrix is

```math
Q_k=
[v_1,\ldots,v_k]
\in\mathbb{R}^{N\times k}
```

The vector

```math
Q_k^\top w
```

measures portfolio exposure to the retained PCA directions.

If

```math
Q_k^\top w\neq0
```

then the portfolio still contains some of the common market-factor movements that PCA identified.

This is undesirable because the trading signal is based on the PCA residual.

---

## Projection onto the Factor Space

Because the columns of $Q_k$ are orthonormal,

```math
Q_k^\top Q_k=I
```

The orthogonal projection matrix onto the PCA factor space is

```math
P_k=Q_kQ_k^\top
```

For any vector $w$,

```math
P_kw
```

is the component of $w$ lying inside the retained PCA factor space.

---

## Projection onto the Residual Space

The orthogonal complement is obtained with

```math
I-P_k
```

Therefore the factor-neutral portfolio is

```math
w^{neutral}
=
(I-Q_kQ_k^\top)w^{raw}
```

This is exactly the same linear-algebra idea used when constructing PCA residual returns.

For returns:

```math
e_t
=
(I-Q_kQ_k^\top)x_t
```

For portfolio weights:

```math
w^{neutral}
=
(I-Q_kQ_k^\top)w^{raw}
```

So both are projections onto the same residual subspace.

---

## Proof of Factor Neutrality

We want to show

```math
Q_k^\top w^{neutral}=0
```

Start from

```math
w^{neutral}
=
(I-Q_kQ_k^\top)w^{raw}
```

Then

```math
Q_k^\top w^{neutral}
=
Q_k^\top
(I-Q_kQ_k^\top)
w^{raw}
```

Expand:

```math
Q_k^\top w^{neutral}
=
Q_k^\top w^{raw}
-
Q_k^\top Q_kQ_k^\top w^{raw}
```

Since

```math
Q_k^\top Q_k=I
```

we get

```math
Q_k^\top w^{neutral}
=
Q_k^\top w^{raw}
-
Q_k^\top w^{raw}
```

Therefore

```math
\boxed{
Q_k^\top w^{neutral}=0
}
```

The final portfolio is orthogonal to every retained PCA factor.

---

## Interpretation as a Hedge

This projection acts as the project's factor hedge.

The final model does not use a separate SPY hedge.

Instead, the portfolio removes exposure directly to the same PCA directions used to define the residual signal.

This is more consistent with the model because the signal itself is defined relative to those PCA factors.

---

## Gross Exposure

Gross exposure is

```math
G
=
\sum_{i=1}^{N}|w_i|
```

It measures total long and short notional without allowing positive and negative positions to cancel.

Example:

```math
w=
(0.20,-0.20)
```

has

```math
G=0.40
```

even though the net exposure is zero.

The final strategy uses

```math
G_{max}=0.50
```

---

## Gross-Exposure Scaling

PCA neutralization may create small non-zero weights in many stocks.

If

```math
G
=
\sum_i|w_i|
```

exceeds the allowed maximum, the entire portfolio is scaled by

```math
\alpha
=
\frac{G_{max}}{G}
```

so that

```math
w^{scaled}
=
\alpha w
```

Then

```math
\sum_i
|w_i^{scaled}|
=
G_{max}
```

Scaling does not destroy PCA neutrality because

```math
Q_k^\top(\alpha w)
=
\alpha Q_k^\top w
=
0
```

---

## Net Exposure

Net exposure is

```math
N
=
\sum_iw_i
```

Example:

```math
w=
(0.20,-0.20)
```

gives

```math
N=0
```

This is dollar neutrality.

However,

```math
Q_k^\top w=0
```

and

```math
\sum_iw_i=0
```

are different conditions.

The final strategy explicitly enforces PCA factor neutrality, not exact dollar neutrality.

---

## Factor Exposure

The function `factor_exposure()` computes

```math
f
=
Q_k^\top w
```

where

```math
f\in\mathbb{R}^{k}
```

Each element of $f$ is the portfolio exposure to one retained principal component.

After neutralization,

```math
f\approx0
```

up to floating-point numerical error.

Typical values in the backtest are around

```math
10^{-16}
```

immediately after neutralization.

---

## Turnover

Turnover measures how much portfolio weight changes between two dates.

For weights $w_{t-1}$ and $w_t$,

```math
\tau_t
=
\sum_i
|w_{t,i}-w_{t-1,i}|
```

Example:

```math
w_{t-1}
=
(0.10,-0.10,0)
```

and

```math
w_t
=
(0,0,0.10)
```

give

```math
\tau_t
=
|0-0.10|
+
|0-(-0.10)|
+
|0.10-0|
```

so

```math
\tau_t=0.30
```

This means traded notional equals 30% of portfolio equity.

---

## Why Absolute Values Are Needed

Without absolute values,

```math
\sum_i
(w_{t,i}-w_{t-1,i})
```

can cancel purchases against sales.

For example,

```math
\Delta w=(+0.10,-0.10)
```

would sum to zero even though 20% of equity was traded.

Therefore turnover uses

```math
\sum_i|\Delta w_i|
```

instead.

---

## Role in the Full Strategy

The portfolio layer performs four operations:

```math
\text{position states}
\rightarrow
\text{raw weights}
\rightarrow
\text{PCA neutralization}
\rightarrow
\text{gross-exposure scaling}
```

The resulting weights are then passed to the chronological backtest.

The central equation is

```math
\boxed{
w_t
=
(I-Q_{k,t}Q_{k,t}^\top)
w_t^{raw}
}
```

subject to the gross-exposure constraint

```math
\sum_i|w_{t,i}|
\leq0.50
```