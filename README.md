# Market Factor Modeling & Statistical Arbitrage

Research project using **PCA** to separate common market movements from stock-specific residuals and test residual mean reversion.

Main notebook:

`notebooks/03_final_research_results.ipynb`

## Idea

Let $x_t \in \mathbb{R}^N$ be the vector of centered stock returns.

The sample covariance matrix is

$$\Sigma=\frac{1}{T-1}X^\top X$$

with eigendecomposition

$$\Sigma=Q\Lambda Q^\top$$

The first $k$ eigenvectors form

$$Q_k=[v_1,\dots,v_k]$$

The common-factor component is

$$\hat{x}_t=Q_kQ_k^\top x_t$$

and the PCA residual is

$$e_t=x_t-\hat{x}_t=(I-Q_kQ_k^\top)x_t$$

Since

$$Q_k^\top e_t=0$$

the residual is orthogonal to the retained PCA factors.

## Signal

Residuals are accumulated over $H$ days:

$$s_t=\sum_{j=0}^{H-1}e_{t-j}$$

Then standardized using past data:

$$z_t=\frac{s_t-\mu_t}{\sigma_t}$$

Trading rule:

- $z<-2.5$ → long candidate
- $z>2.5$ → short candidate

Final model uses $H=40$.

## PC1 Regime Filter

Define the fraction of variance explained by PC1:

$$m_t=\frac{\lambda_{1,t}}{\sum_j\lambda_{j,t}}$$

New trades are allowed only when the current PC1 share is above the median of the previous 252 days:

$$m_t > q_t$$

where $q_t$ is the rolling 252-day median of past PC1 shares.

The goal is to trade residual mean reversion only when the common market-factor structure is relatively strong.

## Factor-Neutral Portfolio

Raw signal weights $w^{raw}$ are projected away from the retained PCA factors:

$$w=(I-Q_kQ_k^\top)w^{raw}$$

Therefore

$$Q_k^\top w=0$$

before gross-exposure scaling.

## Final Parameters

| Parameter | Value |
|---|---:|
| PCA window | 252 days |
| PCs | 6 |
| Residual horizon | 40 days |
| Entry threshold | 2.5 |
| Max holding | 40 days |
| Max positions | 10 |
| Position size | 5% |
| Max gross exposure | 50% |
| Transaction cost | 10 bps |
| Regime window | 252 days |

## Results

| Metric | Result |
|---|---:|
| Cumulative return | 3.50% |
| Annualized volatility | 1.06% |
| Sharpe ratio | 0.59 |
| Max drawdown | -1.54% |
| Average daily turnover | 0.72% |
| Total transaction costs | 1.01% |

Without the PC1 regime filter:

| Metric | Baseline | PC1 Filter |
|---|---:|---:|
| Return | -1.24% | 3.50% |
| Sharpe | -0.13 | 0.59 |
| Max drawdown | -4.49% | -1.54% |
| Daily turnover | 1.69% | 0.72% |

## Robustness

Empirical quantile thresholds and an AR(1) filter were tested but did not improve the strategy.

The main limitation is sensitivity to the residual horizon:

| $H$ | Sharpe |
|---:|---:|
| 30 | -0.03 |
| 40 | 0.59 |
| 50 | 0.05 |
| 60 | -0.03 |

So the result should be treated as **exploratory**, not as evidence of a stable trading alpha.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_pipeline.ipynb
│   ├── 02_residual_diagnostics.ipynb
│   └── 03_final_research_results.ipynb
└── src/
    ├── data.py
    ├── pca.py
    ├── residuals.py
    ├── signals.py
    ├── portfolio.py
    ├── backtest.py
    └── metrics.py
```

## Run

```bash
pip install -r requirements.txt
jupyter lab
```

Open:

```text
notebooks/03_final_research_results.ipynb
```