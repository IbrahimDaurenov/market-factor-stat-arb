# `pca.py`

This module implements PCA directly from the covariance matrix.

The main steps are

```math
X
\rightarrow
X_c
\rightarrow
\Sigma
\rightarrow
Q\Lambda Q^\top
\rightarrow
Q_k
```

---

## Data Matrix

Suppose there are $T$ trading days and $N$ stocks.

The return matrix is

```math
X\in\mathbb{R}^{T\times N}
```

where

- each row is one trading day;
- each column is one stock.

The return vector for day $t$ is

```math
x_t\in\mathbb{R}^{N}
```

---

## Centering

PCA is applied to centered data.

The sample mean vector is

```math
\mu
=
\frac{1}{T}
\sum_{t=1}^{T}x_t
```

with

```math
\mu\in\mathbb{R}^{N}
```

The centered observation is

```math
z_t=x_t-\mu
```

and the centered data matrix is

```math
X_c
=
X-\mathbf{1}\mu^\top
```

where

```math
X_c\in\mathbb{R}^{T\times N}
```

Each column of $X_c$ has sample mean zero.

---

## Sample Covariance Matrix

The covariance matrix is

```math
\Sigma
=
\frac{1}{T-1}
X_c^\top X_c
```

with

```math
\Sigma\in\mathbb{R}^{N\times N}
```

To see why this works, consider entry $(i,j)$:

```math
(X_c^\top X_c)_{ij}
=
\sum_{t=1}^{T}
z_{t,i}z_{t,j}
```

Therefore,

```math
\Sigma_{ij}
=
\frac{1}{T-1}
\sum_{t=1}^{T}
z_{t,i}z_{t,j}
```

which is the sample covariance between stocks $i$ and $j$.

For $i=j$,

```math
\Sigma_{ii}
=
\operatorname{Var}(X_i)
```

so the diagonal contains sample variances.

---

## Symmetry

The covariance matrix is symmetric:

```math
\Sigma^\top=\Sigma
```

because

```math
(X_c^\top X_c)^\top
=
X_c^\top X_c
```

This is important because real symmetric matrices have real eigenvalues and an orthonormal eigenvector basis.

---

## Positive Semidefinite Property

For any vector

```math
a\in\mathbb{R}^{N}
```

we have

```math
a^\top\Sigma a
=
\frac{1}{T-1}
a^\top X_c^\top X_c a
```

and therefore

```math
a^\top\Sigma a
=
\frac{1}{T-1}
\|X_ca\|^2
\geq0
```

Hence $\Sigma$ is positive semidefinite.

Therefore,

```math
\lambda_i\geq0
```

for every covariance eigenvalue.

---

## Eigendecomposition

Because $\Sigma$ is symmetric,

```math
\Sigma
=
Q\Lambda Q^\top
```

where

```math
Q
=
[v_1,\ldots,v_N]
```

contains eigenvectors and

```math
\Lambda
=
\operatorname{diag}
(\lambda_1,\ldots,\lambda_N)
```

contains eigenvalues.

Each eigenvector satisfies

```math
\Sigma v_i
=
\lambda_i v_i
```

and the eigenvectors can be chosen orthonormal:

```math
v_i^\top v_j
=
0
\qquad
(i\neq j)
```

and

```math
\|v_i\|=1
```

Thus,

```math
Q^\top Q=I
```

---

## Variance Along a Direction

Take any unit direction

```math
v\in\mathbb{R}^{N}
```

with

```math
\|v\|=1
```

The scalar coordinate of a centered observation $z$ along $v$ is

```math
c=v^\top z
```

Because the data are centered,

```math
E[c]=0
```

so

```math
\operatorname{Var}(c)
=
E[c^2]
```

Now,

```math
c^2
=
(v^\top z)^2
```

and since a scalar equals its transpose,

```math
(v^\top z)^2
=
v^\top zz^\top v
```

Therefore,

```math
\operatorname{Var}(c)
=
v^\top E[zz^\top]v
```

Since

```math
E[zz^\top]=\Sigma
```

we obtain

```math
\boxed{
\operatorname{Var}(v^\top z)
=
v^\top\Sigma v
}
```

---

## Why PCA Uses Eigenvectors

PCA asks:

> Which unit direction contains the largest possible variance?

Mathematically,

```math
\max_{\|v\|=1}
v^\top\Sigma v
```

Using a Lagrange multiplier,

```math
L(v,\lambda)
=
v^\top\Sigma v
-
\lambda(v^\top v-1)
```

Differentiating with respect to $v$ gives

```math
2\Sigma v
-
2\lambda v
=
0
```

hence

```math
\Sigma v
=
\lambda v
```

So every stationary direction is an eigenvector.

The largest value of

```math
v^\top\Sigma v
```

is obtained at the eigenvector associated with the largest eigenvalue.

Therefore the first principal component direction is

```math
v_1
```

with

```math
\lambda_1
=
\max_{\|v\|=1}
v^\top\Sigma v
```

---

## Eigenvalue as Explained Variance

For eigenvector $v_i$,

```math
\Sigma v_i
=
\lambda_i v_i
```

Then

```math
\operatorname{Var}(v_i^\top z)
=
v_i^\top\Sigma v_i
```

so

```math
\operatorname{Var}(v_i^\top z)
=
v_i^\top
(\lambda_i v_i)
```

and because

```math
v_i^\top v_i=1
```

we get

```math
\boxed{
\operatorname{Var}(v_i^\top z)
=
\lambda_i
}
```

Thus each eigenvalue measures the variance explained by its principal component.

---

## Ordering the Principal Components

`numpy.linalg.eigh()` returns eigenvalues in ascending order.

The code reverses this ordering so that

```math
\lambda_1
\geq
\lambda_2
\geq
\cdots
\geq
\lambda_N
```

The corresponding eigenvectors are reordered in exactly the same way.

---

## Explained Variance Ratio

Total variance equals the trace of the covariance matrix:

```math
\operatorname{tr}(\Sigma)
=
\sum_{i=1}^{N}\lambda_i
```

Therefore the explained-variance ratio of PC $i$ is

```math
EVR_i
=
\frac{\lambda_i}
{\sum_{j=1}^{N}\lambda_j}
```

For the first $k$ principal components,

```math
EVR_{1:k}
=
\frac{
\lambda_1+\cdots+\lambda_k
}{
\sum_{j=1}^{N}\lambda_j
}
```

In code:

```python
explained_variance_ratio = (
    eigenvalues
    / eigenvalues.sum()
)
```

---

## Retaining the First k Components

Let

```math
Q_k
=
[v_1,\ldots,v_k]
```

Then

```math
Q_k\in\mathbb{R}^{N\times k}
```

The project uses

```math
k=6
```

in the final model.

---

## PCA Scores

For one centered observation,

```math
z_t=x_t-\mu
```

The PCA coordinates are

```math
c_t
=
Q_k^\top z_t
```

where

```math
c_t\in\mathbb{R}^{k}
```

The $i$-th score is

```math
c_{t,i}
=
v_i^\top z_t
```

This tells how much principal component $i$ appears on day $t$.

---

## Matrix Form for All Days

Because observations are stored as rows, the score matrix is

```math
C_k
=
X_cQ_k
```

with

```math
C_k\in\mathbb{R}^{T\times k}
```

This is what `transform()` computes.

---

## Loadings vs Scores

These two concepts are different.

An eigenvector

```math
v_i
```

contains the loadings of PC $i$.

For example,

```math
v_1
=
\begin{bmatrix}
v_{1,1}\\
v_{1,2}\\
\vdots\\
v_{1,N}
\end{bmatrix}
```

contains one loading for each stock.

The score

```math
c_{t,i}
=
v_i^\top z_t
```

depends on the trading day.

So:

```math
\text{loading}
=
\text{coordinate of the PCA direction}
```

while

```math
\text{score}
=
\text{amount of that PCA direction on day }t
```

---

## Reconstruction

Using the first $k$ PCA scores,

```math
c_t
=
Q_k^\top z_t
```

the low-rank reconstruction is

```math
\hat z_t
=
Q_kc_t
```

Substituting the score formula,

```math
\hat z_t
=
Q_kQ_k^\top z_t
```

For the full data matrix,

```math
\hat X_c
=
X_cQ_kQ_k^\top
```

This is what `reconstruct()` computes.

---

## Projection Interpretation

Define

```math
P_k
=
Q_kQ_k^\top
```

Then

```math
\hat z_t=P_kz_t
```

Because the columns of $Q_k$ are orthonormal,

```math
P_k^\top=P_k
```

and

```math
P_k^2=P_k
```

so $P_k$ is an orthogonal projection matrix.

Therefore PCA reconstruction is exactly the projection of the return vector onto

```math
\operatorname{span}
(v_1,\ldots,v_k)
```

The remaining component is handled in `residuals.py`.