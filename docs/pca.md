# `pca.py`

This module implements PCA directly from the covariance matrix.

The main steps are

```math
X \rightarrow X_c \rightarrow \Sigma \rightarrow Q\Lambda Q^\top \rightarrow Q_k
```

---

## Data Matrix

Suppose there are $T$ trading days and $N$ stocks.

The return matrix is

```math
X \in \mathbb{R}^{T \times N}
```

Each row is one trading day and each column is one stock.

The return vector for day $t$ is

```math
x_t \in \mathbb{R}^N
```

---

## Centering

PCA is applied to centered data.

The sample mean vector is

```math
\mu = \frac{1}{T}\sum_{t=1}^{T}x_t
```

with

```math
\mu \in \mathbb{R}^N
```

The centered return vector is

```math
z_t = x_t - \mu
```

and the centered data matrix is

```math
X_c = X - \mathbf{1}\mu^\top
```

with

```math
X_c \in \mathbb{R}^{T \times N}
```

Each column of $X_c$ has sample mean zero.

---

## Sample Covariance Matrix

The sample covariance matrix is

```math
\Sigma = \frac{1}{T-1}X_c^\top X_c
```

with

```math
\Sigma \in \mathbb{R}^{N \times N}
```

Entry $(i,j)$ is

```math
\Sigma_{ij}
=
\frac{1}{T-1}
\sum_{t=1}^{T}z_{t,i}z_{t,j}
```

so off-diagonal entries measure covariance between stocks.

The diagonal entries are sample variances:

```math
\Sigma_{ii} = \mathrm{Var}(X_i)
```

---

## Symmetry

The covariance matrix is symmetric:

```math
\Sigma^\top = \Sigma
```

because

```math
(X_c^\top X_c)^\top = X_c^\top X_c
```

A real symmetric matrix has real eigenvalues and an orthonormal eigenvector basis.

---

## Positive Semidefinite Property

For any vector

```math
a \in \mathbb{R}^N
```

we have

```math
a^\top \Sigma a
=
\frac{1}{T-1}
a^\top X_c^\top X_c a
```

Therefore,

```math
a^\top \Sigma a
=
\frac{1}{T-1}
\|X_c a\|^2
\ge 0
```

So $\Sigma$ is positive semidefinite and all covariance eigenvalues satisfy

```math
\lambda_i \ge 0
```

---

## Eigendecomposition

Because $\Sigma$ is real and symmetric,

```math
\Sigma = Q\Lambda Q^\top
```

where

```math
Q = [v_1,\ldots,v_N]
```

contains the eigenvectors.

The eigenvalue matrix is

```math
\Lambda
=
\begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_N
\end{bmatrix}
```

Each eigenvector satisfies

```math
\Sigma v_i = \lambda_i v_i
```

and the eigenvectors are orthonormal:

```math
v_i^\top v_j = 0
\quad
(i \ne j)
```

and

```math
v_i^\top v_i = 1
```

Therefore,

```math
Q^\top Q = I
```

---

## Variance Along a Direction

Take a unit vector

```math
v \in \mathbb{R}^N,
\qquad
v^\top v = 1
```

The coordinate of a centered return vector $z$ along $v$ is

```math
c = v^\top z
```

Since the data are centered,

```math
E[c] = 0
```

so

```math
\mathrm{Var}(c) = E[c^2]
```

Now,

```math
c^2 = (v^\top z)^2
```

and

```math
(v^\top z)^2 = v^\top z z^\top v
```

Therefore,

```math
\mathrm{Var}(c)
=
v^\top E[zz^\top]v
```

Since

```math
E[zz^\top] = \Sigma
```

we obtain

```math
\mathrm{Var}(v^\top z)
=
v^\top \Sigma v
```

This is the key connection between PCA and variance.

---

## Why PCA Uses Eigenvectors

PCA looks for the unit direction with maximum variance:

```math
\max_{v^\top v=1} v^\top \Sigma v
```

Using a Lagrange multiplier,

```math
L(v,\lambda)
=
v^\top \Sigma v
-
\lambda(v^\top v-1)
```

Differentiating with respect to $v$ gives

```math
2\Sigma v - 2\lambda v = 0
```

so

```math
\Sigma v = \lambda v
```

Therefore the stationary directions are eigenvectors of the covariance matrix.

The direction with the largest variance is the eigenvector corresponding to the largest eigenvalue:

```math
\Sigma v_1 = \lambda_1 v_1
```

with

```math
\lambda_1
=
\max_{v^\top v=1}
v^\top \Sigma v
```

So $v_1$ is the first principal component direction.

---

## Eigenvalues and Explained Variance

For an eigenvector $v_i$,

```math
\Sigma v_i = \lambda_i v_i
```

The variance along this direction is

```math
\mathrm{Var}(v_i^\top z)
=
v_i^\top \Sigma v_i
```

Using the eigenvalue equation,

```math
\mathrm{Var}(v_i^\top z)
=
v_i^\top (\lambda_i v_i)
```

and since

```math
v_i^\top v_i = 1
```

we get

```math
\mathrm{Var}(v_i^\top z)
=
\lambda_i
```

Therefore each eigenvalue is the variance explained by its principal component.

---

## Ordering the Principal Components

`numpy.linalg.eigh()` returns eigenvalues in ascending order.

The code reverses them so that

```math
\lambda_1
\ge
\lambda_2
\ge
\cdots
\ge
\lambda_N
```

The eigenvectors are reordered in the same way.

---

## Explained Variance Ratio

The total variance is the trace of the covariance matrix:

```math
\mathrm{tr}(\Sigma)
=
\sum_{i=1}^{N}\lambda_i
```

The explained-variance ratio of principal component $i$ is

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
\frac{\lambda_1+\cdots+\lambda_k}
{\sum_{j=1}^{N}\lambda_j}
```

In code:

```python
explained_variance_ratio = (
    eigenvalues
    / eigenvalues.sum()
)
```

The final strategy uses

```math
k = 6
```

principal components.

---

## Retained PCA Basis

Let

```math
Q_k = [v_1,\ldots,v_k]
```

Then

```math
Q_k \in \mathbb{R}^{N \times k}
```

and

```math
Q_k^\top Q_k = I
```

The columns of $Q_k$ span the retained PCA factor subspace inside $\mathbb{R}^N$.

---

## PCA Scores

For a centered return vector

```math
z_t = x_t-\mu
```

the PCA coordinates are

```math
c_t = Q_k^\top z_t
```

with

```math
c_t \in \mathbb{R}^k
```

The individual score for PC $i$ is

```math
c_{t,i} = v_i^\top z_t
```

A score measures how strongly a particular PCA direction appears on day $t$.

---

## Matrix Form

The observations are stored as rows.

Therefore the score matrix for all days is

```math
C_k = X_c Q_k
```

with

```math
C_k \in \mathbb{R}^{T \times k}
```

This is what `transform()` computes.

---

## Loadings and Scores

A loading is a coordinate of an eigenvector.

For example,

```math
v_1
=
\begin{bmatrix}
v_{1,1} \\
v_{1,2} \\
\vdots \\
v_{1,N}
\end{bmatrix}
```

contains the loadings of PC1 across the $N$ stocks.

These values describe the PCA direction itself.

A score is different:

```math
c_{t,i} = v_i^\top z_t
```

It depends on the trading day and measures the amount of PC $i$ present in that observation.

So:

- loadings describe the PCA direction;
- scores describe the position of an observation along that direction.

---

## Reconstruction

Using the first $k$ PCA scores,

```math
c_t = Q_k^\top z_t
```

the low-rank reconstruction is

```math
\hat z_t = Q_k c_t
```

Substituting the score equation gives

```math
\hat z_t
=
Q_k Q_k^\top z_t
```

For the full data matrix,

```math
\hat X_c
=
X_c Q_k Q_k^\top
```

This is what `reconstruct()` computes.

---

## Projection Interpretation

Define

```math
P_k = Q_k Q_k^\top
```

Then

```math
\hat z_t = P_k z_t
```

Because the columns of $Q_k$ are orthonormal,

```math
P_k^\top = P_k
```

and

```math
P_k^2 = P_k
```

Therefore $P_k$ is an orthogonal projection matrix.

The PCA reconstruction is the projection of $z_t$ onto the subspace generated by

```math
v_1,\ldots,v_k
```

The remaining component lies in the orthogonal complement of that PCA subspace and is computed in `residuals.py`.
