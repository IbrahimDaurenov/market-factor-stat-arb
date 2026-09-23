# `data.py`

This module downloads prices and converts them into daily returns.

## Daily return

For stock $i$, let $P_{t,i}$ be its closing price on day $t$.

```math
r_{t,i}=\frac{P_{t,i}}{P_{t-1,i}}-1
```

For $N$ stocks, one trading day is a vector

```math
x_t=
\begin{bmatrix}
r_{t,1}\\
r_{t,2}\\
\vdots\\
r_{t,N}
\end{bmatrix}
\in\mathbb{R}^N
```

Stacking $T$ trading days gives

```math
X=
\begin{bmatrix}
x_1^\top\\
x_2^\top\\
\vdots\\
x_T^\top
\end{bmatrix}
\in\mathbb{R}^{T\times N}
```

So:

- each row is one trading day;
- each column is one stock;
- `stock_returns` is the matrix $X$ used by the rest of the project.

The final strategy does not use a separate SPY hedge. Market-factor exposure is removed later using PCA projection.