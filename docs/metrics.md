# `metrics.py`

This module evaluates the backtest.

The main quantities are:

- equity curve;
- cumulative return;
- annualized volatility;
- Sharpe ratio;
- drawdown;
- turnover and exposure statistics.

---

## Equity Curve

Let $R_t$ be the portfolio return during period $t$.

Starting with capital $V_0$, portfolio value evolves as

```math
V_t=V_{t-1}(1+R_t)
```

Therefore,

```math
V_T
=
V_0
\prod_{t=1}^{T}(1+R_t)
```

In code:

```python
growth = (1 + returns).cumprod()
equity = initial_capital * growth
```

---

## Cumulative Return

Total compounded return is

```math
R_{\mathrm{cum}}
=
\prod_{t=1}^{T}(1+R_t)-1
```

This is different from simply adding daily returns because investment growth compounds through time.

For example, if

```math
R_1=0.10,\qquad R_2=-0.10
```

then

```math
(1.10)(0.90)-1=-0.01
```

so the total return is $-1\%$, not zero.

---

## Annualized Volatility

Let the sample standard deviation of daily returns be

```math
\sigma_d
=
\sqrt{
\frac{1}{T-1}
\sum_{t=1}^{T}
(R_t-\bar R)^2
}
```

Assuming approximately 252 trading days per year,

```math
\sigma_{\mathrm{annual}}
=
\sqrt{252}\,\sigma_d
```

The square-root scaling comes from variance additivity.

If daily returns were independent,

```math
\operatorname{Var}
\left(
\sum_{t=1}^{252}R_t
\right)
=
252\sigma_d^2
```

and therefore

```math
\sigma_{\mathrm{annual}}
=
\sqrt{252\sigma_d^2}
=
\sqrt{252}\sigma_d
```

---

## Sharpe Ratio

The Sharpe ratio measures average excess return relative to return volatility.

Let the annual risk-free rate be $r_f$.

The daily approximation is

```math
r_{f,d}
=
\frac{r_f}{252}
```

Daily excess return is

```math
R_t^{excess}
=
R_t-r_{f,d}
```

The annualized Sharpe ratio is

```math
\mathrm{Sharpe}
=
\sqrt{252}
\frac{
\overline{R^{excess}}
}{
\sigma(R^{excess})
}
```

The project uses

```math
r_f=0
```

so this becomes approximately

```math
\mathrm{Sharpe}
=
\sqrt{252}
\frac{\bar R}{\sigma_R}
```

A larger positive Sharpe means more average return per unit of volatility.

---

## Drawdown

Let $V_t$ be portfolio value.

Define the running historical peak

```math
M_t
=
\max_{s\leq t}V_s
```

The drawdown at time $t$ is

```math
D_t
=
\frac{V_t-M_t}{M_t}
```

Since

```math
V_t\leq M_t
```

drawdown is always non-positive.

If the portfolio falls from

```math
100000
```

to

```math
95000
```

then

```math
D
=
\frac{95000-100000}{100000}
=
-0.05
```

so the drawdown is $-5\%$.

---

## Maximum Drawdown

Maximum drawdown is the worst drawdown observed during the backtest:

```math
MDD
=
\min_t D_t
```

For example,

```math
D_t
=
(-0.01,-0.03,-0.02,-0.07,-0.04)
```

gives

```math
MDD=-0.07
```

or $-7\%$.

---

## Average Daily Turnover

The backtest defines turnover as

```math
\tau_t
=
\sum_i
|w_{t,i}-w_{t-1,i}|
```

The reported average daily turnover is

```math
\bar\tau
=
\frac{1}{T}
\sum_{t=1}^{T}\tau_t
```

Higher turnover generally means higher trading costs.

---

## Total Transaction Costs

If transaction cost on day $t$ is $C_t$, total backtest cost is

```math
C_{\mathrm{total}}
=
\sum_{t=1}^{T}C_t
```

These costs are already subtracted when constructing `net_return`.

---

## Average Gross Exposure

Gross exposure is

```math
G_t
=
\sum_i|w_{t,i}|
```

The reported statistic is

```math
\bar G
=
\frac{1}{T}
\sum_{t=1}^{T}G_t
```

This measures how much of the available portfolio capacity is used on average.

For example,

```math
G_t=0.20
```

means gross positions equal 20% of portfolio capital.

---

## Average Active Positions

Let

```math
A_t
=
\sum_i
\mathbf{1}(p_{t,i}\neq0)
```

be the number of active stock positions on day $t$.

The reported value is

```math
\bar A
=
\frac{1}{T}
\sum_{t=1}^{T}A_t
```

This measures how concentrated or sparse the strategy is through time.

---

## Performance Summary

`performance_summary()` combines the main statistics:

```text
Cumulative Return
Annualized Volatility
Sharpe Ratio
Max Drawdown
Average Daily Turnover
Total Transaction Costs
Average Gross Exposure
Average Active Positions
```

The function evaluates `net_return`, so transaction costs are included in the main performance results.