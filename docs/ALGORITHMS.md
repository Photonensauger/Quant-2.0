# Quant 2.0 -- Mathematical Algorithms Reference

> Complete mathematical specification of every algorithm implemented in the
> Quant 2.0 quantitative trading system.  Formulas use LaTeX notation rendered
> by GitHub / MathJax.

---

## Table of Contents

1. [Feature Engineering](#1-feature-engineering)
   - 1.1 [Technical Indicators](#11-technical-indicators)
   - 1.2 [Cyclical Time Features](#12-cyclical-time-features)
   - 1.3 [Rolling Z-Score Normalisation](#13-rolling-z-score-normalisation)
   - 1.4 [Correlation Filter](#14-correlation-filter)
2. [Bayesian Online Changepoint Detection (BOCPD)](#2-bayesian-online-changepoint-detection-bocpd)
3. [Deep Learning Architectures](#3-deep-learning-architectures)
   - 3.1 [Decoder-Only Transformer](#31-decoder-only-transformer)
   - 3.2 [iTransformer (Inverted Transformer)](#32-itransformer-inverted-transformer)
   - 3.3 [AttentionLSTM](#33-attentionlstm)
   - 3.4 [MomentumTransformer](#34-momentumtransformer)
4. [Loss Functions](#4-loss-functions)
   - 4.1 [Combined Loss (MSE + Directional)](#41-combined-loss-mse--directional)
   - 4.2 [Differentiable Sharpe Ratio](#42-differentiable-sharpe-ratio)
5. [PPO for Portfolio Optimisation](#5-ppo-for-portfolio-optimisation)
6. [Walk-Forward Cross-Validation](#6-walk-forward-cross-validation)
7. [Position Sizing](#7-position-sizing)
   - 7.1 [Kelly Criterion](#71-kelly-criterion)
   - 7.2 [Volatility-Adjusted (ATR-Based)](#72-volatility-adjusted-atr-based)
   - 7.3 [Fixed Fraction](#73-fixed-fraction)
8. [Portfolio Optimisation](#8-portfolio-optimisation)
   - 8.1 [Mean-Variance](#81-mean-variance)
   - 8.2 [Minimum Variance](#82-minimum-variance)
   - 8.3 [Risk Parity](#83-risk-parity)
9. [Risk Management](#9-risk-management)
   - 9.1 [Value at Risk (VaR 95%)](#91-value-at-risk-var-95)
   - 9.2 [Maximum Drawdown](#92-maximum-drawdown)
   - 9.3 [Daily Loss Limit](#93-daily-loss-limit)
   - 9.4 [Position Concentration Limit](#94-position-concentration-limit)
   - 9.5 [ATR-Based Stop-Loss and Take-Profit](#95-atr-based-stop-loss-and-take-profit)
10. [Ensemble Methods](#10-ensemble-methods)
    - 10.1 [Sharpe-Weighted Voting](#101-sharpe-weighted-voting)
    - 10.2 [Confidence Aggregation](#102-confidence-aggregation)
11. [Performance Metrics](#11-performance-metrics)

---

## 1. Feature Engineering

**Source:** `quant/features/technical.py`, `quant/features/time_features.py`, `quant/features/pipeline.py`

The feature pipeline transforms raw OHLCV bars into a normalised, decorrelated
feature matrix suitable for neural network consumption.  The overall flow is:

```
OHLCV --> Technical Indicators (41) --> Time Features (10)
      --> BOCPD Features (2) --> NaN/Correlation Filter --> Rolling Z-Score
      --> aligned (features, targets) numpy arrays
```

### 1.1 Technical Indicators

All indicators operate on the standard OHLCV columns
$(O_t, H_t, L_t, C_t, V_t)$.  The system computes 41 features grouped
into six families.

#### 1.1.1 Returns and Trend

**Log return:**

$$r_t = \ln\!\left(\frac{C_t}{C_{t-1}}\right)$$

**SMA distance** (periods $n \in \{20, 50\}$):

$$\text{SMA}(n)_t = \frac{1}{n}\sum_{i=0}^{n-1} C_{t-i}$$

$$\text{sma\_dist}(n)_t = \frac{C_t - \text{SMA}(n)_t}{\text{SMA}(n)_t + \epsilon}$$

**EMA distance** (spans $s \in \{12, 26\}$):

$$\text{EMA}(s)_t = \alpha \cdot C_t + (1 - \alpha)\cdot\text{EMA}(s)_{t-1}, \quad \alpha = \frac{2}{s+1}$$

$$\text{ema\_dist}(s)_t = \frac{C_t - \text{EMA}(s)_t}{\text{EMA}(s)_t + \epsilon}$$

#### 1.1.2 MACD

$$\text{MACD}_t = \text{EMA}(12)_t - \text{EMA}(26)_t$$

$$\text{Signal}_t = \text{EMA}(9 \mid \text{MACD}_t)$$

$$\text{Histogram}_t = \text{MACD}_t - \text{Signal}_t$$

#### 1.1.3 RSI (Relative Strength Index)

For periods $n \in \{14, 28\}$:

$$\Delta_t = C_t - C_{t-1}$$

$$\text{Gain}_t = \max(\Delta_t, 0), \quad \text{Loss}_t = \max(-\Delta_t, 0)$$

$$\overline{\text{Gain}}_t = \text{EWM}(\text{Gain}, \text{com}=n-1), \quad \overline{\text{Loss}}_t = \text{EWM}(\text{Loss}, \text{com}=n-1)$$

$$\text{RS}_t = \frac{\overline{\text{Gain}}_t}{\overline{\text{Loss}}_t + \epsilon}$$

$$\text{RSI}(n)_t = 100 - \frac{100}{1 + \text{RS}_t}$$

#### 1.1.4 Bollinger Bands

$$\mu_t = \text{SMA}(20)_t, \quad \sigma_t = \text{Std}(C, 20)_t$$

$$\text{Upper}_t = \mu_t + 2\sigma_t, \quad \text{Lower}_t = \mu_t - 2\sigma_t$$

$$\text{BB\_width}_t = \frac{\text{Upper}_t - \text{Lower}_t}{\mu_t + \epsilon}$$

$$\text{BB\_pct}_t = \frac{C_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t + \epsilon}$$

$$\text{BB\_mid\_dist}_t = \frac{C_t - \mu_t}{\mu_t + \epsilon}$$

#### 1.1.5 Average True Range (ATR)

$$\text{TR}_t = \max\bigl(H_t - L_t,\; |H_t - C_{t-1}|,\; |L_t - C_{t-1}|\bigr)$$

For periods $n \in \{14, 28\}$:

$$\text{ATR}(n)_t = \text{EWM}(\text{TR}, \text{span}=n)_t$$

Features are normalised by price:

$$\text{atr\_feat}(n)_t = \frac{\text{ATR}(n)_t}{C_t + \epsilon}$$

#### 1.1.6 On-Balance Volume (OBV)

$$\text{OBV}_t = \sum_{i=1}^{t} \text{sgn}(C_i - C_{i-1}) \cdot V_i$$

Normalised to a rolling z-score:

$$\text{OBV\_zscore}_t = \frac{\text{OBV}_t - \bar{\mu}_{50,t}}{\bar{\sigma}_{50,t} + \epsilon}$$

where $\bar{\mu}_{50,t}$ and $\bar{\sigma}_{50,t}$ are the rolling 50-period mean and standard deviation of OBV.

#### 1.1.7 ADX / Directional Indicators

$$+\text{DM}_t = \max(H_t - H_{t-1}, 0), \quad -\text{DM}_t = \max(L_{t-1} - L_t, 0)$$

(The smaller of the two is zeroed out at each step.)

$$+\text{DI}_t = \frac{100 \cdot \text{EMA}(+\text{DM}, 14)_t}{\text{ATR}(14)_t + \epsilon}$$

$$-\text{DI}_t = \frac{100 \cdot \text{EMA}(-\text{DM}, 14)_t}{\text{ATR}(14)_t + \epsilon}$$

$$\text{DX}_t = \frac{100 \cdot |+\text{DI}_t - (-\text{DI}_t)|}{+\text{DI}_t + (-\text{DI}_t) + \epsilon}$$

$$\text{ADX}_t = \text{EMA}(\text{DX}, 14)_t$$

$$\text{ADXR}_t = \frac{\text{ADX}_t + \text{ADX}_{t-14}}{2}$$

#### 1.1.8 Stochastic Oscillator

$$\%K_t = \frac{100\,(C_t - L_{14,t})}{H_{14,t} - L_{14,t} + \epsilon}$$

where $L_{14,t} = \min_{i \in [t-13, t]} L_i$ and $H_{14,t} = \max_{i \in [t-13, t]} H_i$.

$$\%D_t = \text{SMA}(\%K, 3)_t, \quad \text{Slow}\%D_t = \text{SMA}(\%D, 3)_t$$

#### 1.1.9 Williams %R

$$\%R_t = \frac{-100\,(H_{14,t} - C_t)}{H_{14,t} - L_{14,t} + \epsilon}$$

#### 1.1.10 Commodity Channel Index (CCI)

$$\text{TP}_t = \frac{H_t + L_t + C_t}{3}$$

$$\text{MAD}_{20,t} = \frac{1}{20}\sum_{i=0}^{19} \bigl|\text{TP}_{t-i} - \overline{\text{TP}}_{20,t}\bigr|$$

$$\text{CCI}_t = \frac{\text{TP}_t - \overline{\text{TP}}_{20,t}}{0.015 \cdot \text{MAD}_{20,t} + \epsilon}$$

#### 1.1.11 Money Flow Index (MFI)

$$\text{MF}_t = \text{TP}_t \cdot V_t$$

$$\text{MFI}_t = 100 - \frac{100}{1 + \frac{\sum_{i=0}^{13} \text{MF}^+_{t-i}}{\sum_{i=0}^{13} \text{MF}^-_{t-i} + \epsilon}}$$

where $\text{MF}^+$ accumulates when $\text{TP}_t > \text{TP}_{t-1}$ and $\text{MF}^-$ otherwise.

#### 1.1.12 Rate of Change (ROC) and Momentum

For periods $n \in \{5, 10, 20\}$:

$$\text{ROC}(n)_t = \frac{C_t - C_{t-n}}{C_{t-n}}$$

For periods $n \in \{10, 20\}$:

$$\text{Momentum}(n)_t = C_t - C_{t-n}$$

#### 1.1.13 Realised Volatility

$$\sigma(n)_t = \text{Std}\bigl(\{r_{t-i}\}_{i=0}^{n-1}\bigr), \quad n \in \{10, 20\}$$

#### 1.1.14 Volume-Weighted Average Price (VWAP) Distance

$$\text{VWAP}_{20,t} = \frac{\sum_{i=0}^{19} \text{TP}_{t-i} \cdot V_{t-i}}{\sum_{i=0}^{19} V_{t-i} + \epsilon}$$

$$\text{vwap\_dist}_t = \frac{C_t - \text{VWAP}_{20,t}}{\text{VWAP}_{20,t} + \epsilon}$$

#### 1.1.15 Additional Features

| Feature | Formula |
|---------|---------|
| Volume ratio | $V_t / \text{SMA}(V, 20)_t$ |
| Volume change | $\Delta V_t / V_{t-1}$ |
| Donchian width | $(H_{20}^{\max} - L_{20}^{\min}) / (C_t + \epsilon)$ |
| Donchian %position | $(C_t - L_{20}^{\min}) / (H_{20}^{\max} - L_{20}^{\min} + \epsilon)$ |
| Keltner width | $(\text{KC}^{\text{up}} - \text{KC}^{\text{low}}) / (\text{EMA}(C,20) + \epsilon)$, where KC offset $= 1.5 \times \text{ATR}(20)$ |
| Chaikin Money Flow | $\text{CMF}_{20} = \frac{\sum_{i=0}^{19}\text{CLV}_{t-i}\cdot V_{t-i}}{\sum V_{t-i} + \epsilon}$, $\text{CLV} = \frac{(C-L)-(H-C)}{H-L+\epsilon}$ |
| Price acceleration | $r_t - r_{t-1}$ (second-order momentum) |

### 1.2 Cyclical Time Features

**Source:** `quant/features/time_features.py`

All temporal dimensions are encoded as $(sin, cos)$ pairs to preserve
wrap-around periodicity. For a raw value $v$ with period $P$:

$$\text{sin\_enc}(v, P) = \sin\!\left(\frac{2\pi v}{P}\right)$$

$$\text{cos\_enc}(v, P) = \cos\!\left(\frac{2\pi v}{P}\right)$$

| Temporal Dimension | $v$ | $P$ |
|--------------------|-----|-----|
| Hour of day | `hour + minute/60` | 24 |
| Day of week | `dayofweek` (Mon=0) | 7 |
| Day of month | `day` | 31 |
| Month of year | `month` | 12 |
| Quarter | `quarter` | 4 |

This produces 10 NaN-free features (5 sin + 5 cos).

### 1.3 Rolling Z-Score Normalisation

**Source:** `quant/features/pipeline.py` -- `FeaturePipeline._rolling_zscore_fit`

Each feature column $f$ is normalised using a rolling window of size $W$
(default 252):

$$z_{f,t} = \frac{x_{f,t} - \bar{\mu}_{f,t}}{\bar{\sigma}_{f,t} + \epsilon}$$

where:

$$\bar{\mu}_{f,t} = \frac{1}{\min(t, W)} \sum_{i=\max(1,t-W+1)}^{t} x_{f,i}$$

$$\bar{\sigma}_{f,t} = \sqrt{\frac{1}{\min(t, W)-1} \sum_{i=\max(1,t-W+1)}^{t} (x_{f,i} - \bar{\mu}_{f,t})^2}$$

During inference (`transform`), if the input is shorter than $W$ bars, the
last saved rolling statistics from `fit_transform` are used directly as
$\bar{\mu}$ and $\bar{\sigma}$.

**Configuration:** `FeatureConfig.rolling_window = 252`, `FeatureConfig.epsilon = 1e-8`

### 1.4 Correlation Filter

**Source:** `quant/features/pipeline.py` -- `FeaturePipeline._correlation_filter`

After computing raw features, a pairwise Pearson correlation matrix is formed.
For each pair $(f_i, f_j)$ with $i < j$:

$$|\rho(f_i, f_j)| > \tau \implies \text{drop } f_j$$

where $\tau$ is the correlation threshold (default 0.95).  The upper-triangular
scan ensures deterministic drop order: for any highly correlated pair, the
column appearing later in the DataFrame is dropped.

**Configuration:** `FeatureConfig.correlation_threshold = 0.95`

---

## 2. Bayesian Online Changepoint Detection (BOCPD)

**Source:** `quant/features/changepoint.py`

**Reference:** Adams & MacKay (2007), "Bayesian Online Changepoint Detection"

The BOCPD algorithm detects abrupt changes in the statistical properties of a
streaming time series.  It maintains a posterior distribution over the *run
length* $r_t$ (number of observations since the last changepoint) and updates
it analytically using a Normal-Inverse-Gamma (NIG) conjugate prior.

### 2.1 Generative Model

At each time step $t$, observation $x_t$ is drawn from a segment with
parameters $(\mu, \sigma^2)$ that persist until a changepoint occurs.  The
hazard function gives the prior probability of a changepoint:

$$H(r) = \frac{1}{\lambda}$$

where $\lambda$ is the expected run length (default 200).

### 2.2 Prior and Sufficient Statistics

The NIG prior has hyperparameters $(\mu_0, \kappa_0, \alpha_0, \beta_0)$.
For each run length $r$, the sufficient statistics are updated as:

$$\mu_r' = \frac{\kappa_r \cdot \mu_r + x_t}{\kappa_r + 1}$$

$$\kappa_r' = \kappa_r + 1$$

$$\alpha_r' = \alpha_r + \frac{1}{2}$$

$$\beta_r' = \beta_r + \frac{\kappa_r \cdot (x_t - \mu_r)^2}{2(\kappa_r + 1)}$$

A new run ($r = 0$) is initialised with the prior hyperparameters
$(\mu_0, \kappa_0, \alpha_0, \beta_0)$.

### 2.3 Posterior Predictive Distribution

The posterior predictive under the NIG model is a Student-$t$ distribution
with $2\alpha$ degrees of freedom:

$$p(x_t \mid r) = t_{2\alpha_r}\!\left(\mu_r,\; \frac{\beta_r(\kappa_r + 1)}{\alpha_r \cdot \kappa_r}\right)$$

Its log-density at $x_t$ is:

$$\ln p = \ln\Gamma\!\left(\frac{\nu + 1}{2}\right) - \ln\Gamma\!\left(\frac{\nu}{2}\right) - \frac{1}{2}\ln(\nu\pi) - \ln s - \frac{\nu+1}{2}\ln\!\left(1 + \frac{z^2}{\nu}\right)$$

where $\nu = 2\alpha_r$, $s = \sqrt{\beta_r(\kappa_r + 1)/(\alpha_r \kappa_r)}$, and $z = (x_t - \mu_r)/s$.

### 2.4 Run-Length Posterior Update

**Growth probabilities** (existing run continues):

$$P(r_t = r_{t-1} + 1, x_{1:t}) = P(r_{t-1}, x_{1:t-1}) \cdot p(x_t \mid r_{t-1}) \cdot (1 - H)$$

**Changepoint probability** (new run starts):

$$P(r_t = 0, x_{1:t}) = \sum_{r_{t-1}} P(r_{t-1}, x_{1:t-1}) \cdot p(x_t \mid r_{t-1}) \cdot H$$

The joint is normalised by the evidence:

$$P(r_t \mid x_{1:t}) = \frac{P(r_t, x_{1:t})}{\sum_r P(r_t = r, x_{1:t})}$$

### 2.5 Outputs

| Output | Formula |
|--------|---------|
| Changepoint score | $\text{cp\_score}_t = P(r_t = 0 \mid x_{1:t}) \in [0, 1]$ |
| Severity | $\text{severity}_t = \lvert z_t \rvert$ under the Student-$t$ of the highest-posterior non-zero run |

### 2.6 Truncation

To bound memory, the run-length vector is truncated to $3\lambda$ entries.
After truncation the posterior is renormalised.

**Configuration defaults:** $\lambda = 200$, $\alpha_0 = 1$, $\beta_0 = 1$, $\kappa_0 = 1$, $\mu_0 = 0$

---

## 3. Deep Learning Architectures

All models share the interface:

- **Input:** $\mathbf{X} \in \mathbb{R}^{B \times T \times N}$ (batch, sequence length, features)
- **Output:** $\hat{\mathbf{y}} \in \mathbb{R}^{B \times H}$ (batch, forecast horizon) or $\mathbb{R}^{B \times 1}$ for position models

**Weight initialisation:** Xavier uniform for all weight matrices with $\text{dim} > 1$; orthogonal for LSTM weights.

### 3.1 Decoder-Only Transformer

**Source:** `quant/models/transformer.py` -- `DecoderTransformer`

A causal (autoregressive) Transformer that predicts future log-returns from
the last token's representation.

#### Architecture

```
Input [B, T, N] --> Linear(N, d_model) --> + Positional Encoding
                                            |
                                     L x CausalDecoderLayer
                                            |
                                       LayerNorm
                                            |
                                     h[:, -1, :] (last token)
                                            |
                                   Linear(d_model, H) --> Output [B, H]
```

#### Sinusoidal Positional Encoding

Following Vaswani et al. (2017):

$$\text{PE}(pos, 2k) = \sin\!\left(\frac{pos}{10000^{2k/d_{\text{model}}}}\right)$$

$$\text{PE}(pos, 2k+1) = \cos\!\left(\frac{pos}{10000^{2k/d_{\text{model}}}}\right)$$

The encoding is added to the projected input and followed by dropout.

#### Causal Self-Attention

Each decoder layer uses pre-norm architecture with masked multi-head
self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

where $M$ is an upper-triangular mask with $-\infty$ entries above the
diagonal, ensuring position $t$ can only attend to positions $\leq t$:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

#### Feed-Forward Network

$$\text{FFN}(x) = W_2\;\text{GELU}(W_1 x + b_1) + b_2$$

with $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ and $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$.

#### Pre-Norm Residual

$$x' = x + \text{Dropout}\bigl(\text{Attn}(\text{LN}(x))\bigr)$$

$$x'' = x' + \text{FFN}\bigl(\text{LN}(x')\bigr)$$

**Default configuration:** $d_{\text{model}} = 64$, $n_{\text{heads}} = 4$, $n_{\text{layers}} = 3$, $d_{ff} = 256$, dropout $= 0.1$

### 3.2 iTransformer (Inverted Transformer)

**Source:** `quant/models/itransformer.py`

**Reference:** Liu et al., "iTransformer: Inverted Transformers Are Effective
for Time Series Forecasting" (ICLR 2024)

The key insight is to *transpose* the input so that each feature becomes a
token and the temporal dimension becomes the embedding, enabling the
self-attention mechanism to model cross-variate dependencies rather than
temporal dependencies.

#### Architecture

```
Input [B, T, N] --> Transpose --> [B, N, T]
                                    |
                          Linear(T, d_model)  (per-feature temporal projection)
                                    |
                           + Learnable Feature Embedding
                                    |
                             L x EncoderLayer  (full attention over N feature tokens)
                                    |
                               LayerNorm
                                    |
                         Linear(d_model, H)  (per-feature output)
                                    |
                         [B, N, H] --> Transpose --> [B, H, N]
                                    |
                             Linear(N, 1)  (aggregate features)
                                    |
                              Output [B, H]
```

#### Temporal Projection

Each feature $f_i$ has its time series mapped from length $T$ to $d_{\text{model}}$:

$$\mathbf{h}_i = W_{\text{temp}} \cdot \mathbf{x}^{(i)} + b_{\text{temp}}, \quad \mathbf{x}^{(i)} \in \mathbb{R}^T, \; \mathbf{h}_i \in \mathbb{R}^{d_{\text{model}}}$$

A learnable feature positional embedding $\mathbf{E}_f \in \mathbb{R}^{1 \times N \times d_{\text{model}}}$ (initialised $\sim \mathcal{N}(0, 0.02)$) is added.

#### Encoder Layers (Full Attention)

Unlike the decoder transformer, the encoder uses **full (bidirectional)
attention** over the $N$ feature tokens -- there is no causal mask:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

This allows every feature to attend to every other feature, capturing
cross-variate correlations.

#### Output Aggregation

Per-feature outputs are mapped to $H$ dimensions and aggregated across the
$N$ features via a learned linear layer:

$$\hat{y}_h = \sum_{i=1}^{N} w_i \cdot o_{i,h}$$

**Default configuration:** $d_{\text{model}} = 64$, $n_{\text{heads}} = 4$, $n_{\text{layers}} = 2$, $d_{ff} = 256$, dropout $= 0.1$

### 3.3 AttentionLSTM

**Source:** `quant/models/lstm.py`

A multi-layer LSTM encoder augmented with multi-head self-attention over the
full sequence of hidden states, addressing the information compression
problem of pure recurrent architectures.

#### Architecture

```
Input [B, T, N] --> LSTM(n_features, hidden, n_layers)
                         |
                    [B, T, hidden]  (all hidden states)
                         |
                 MultiHeadAttention(hidden, n_heads)
                         |
              Residual + LayerNorm
                         |
                    h[:, -1, :]  (last attended state)
                         |
              Linear(hidden, hidden) --> GELU --> Dropout
                         |
              Linear(hidden, H) --> Output [B, H]
```

#### LSTM Equations

For each layer $l$ and time step $t$:

$$f_t^{(l)} = \sigma\!\left(W_f^{(l)} h_{t-1}^{(l)} + U_f^{(l)} x_t^{(l)} + b_f^{(l)}\right)$$

$$i_t^{(l)} = \sigma\!\left(W_i^{(l)} h_{t-1}^{(l)} + U_i^{(l)} x_t^{(l)} + b_i^{(l)}\right)$$

$$o_t^{(l)} = \sigma\!\left(W_o^{(l)} h_{t-1}^{(l)} + U_o^{(l)} x_t^{(l)} + b_o^{(l)}\right)$$

$$\tilde{c}_t^{(l)} = \tanh\!\left(W_c^{(l)} h_{t-1}^{(l)} + U_c^{(l)} x_t^{(l)} + b_c^{(l)}\right)$$

$$c_t^{(l)} = f_t^{(l)} \odot c_{t-1}^{(l)} + i_t^{(l)} \odot \tilde{c}_t^{(l)}$$

$$h_t^{(l)} = o_t^{(l)} \odot \tanh(c_t^{(l)})$$

where $\sigma$ is the sigmoid function and $\odot$ denotes element-wise
multiplication.  LSTM weights are initialised with orthogonal initialisation.

#### Self-Attention over Hidden States

The full sequence of LSTM hidden states $\mathbf{H} = [h_1, h_2, \ldots, h_T] \in \mathbb{R}^{B \times T \times d_h}$ is passed through multi-head self-attention:

$$\mathbf{A} = \text{MultiHead}(\mathbf{H}, \mathbf{H}, \mathbf{H})$$

$$\mathbf{O} = \text{LayerNorm}(\mathbf{H} + \mathbf{A})$$

The last time-step $\mathbf{O}[:, -1, :]$ is used as the context vector for
the output head.

**Default configuration:** hidden $= 128$, layers $= 2$, dropout $= 0.2$, attention heads $= 4$

### 3.4 MomentumTransformer

**Source:** `quant/models/momentum_transformer.py`

A variant of the decoder-only Transformer that directly outputs a continuous
position signal in $[-1, +1]$ (via tanh) suitable for momentum/trend-following
strategies.

#### Architecture

Identical to the DecoderTransformer (Section 3.1) except the output head:

```
Last token representation [B, d_model]
        |
  Linear(d_model, d_model/2) --> GELU --> Dropout
        |
  Linear(d_model/2, 1) --> Tanh
        |
  Output [B, 1]  in [-1, +1]
```

The output represents:
- $+1$ = maximum long exposure
- $-1$ = maximum short exposure
- $0$ = flat (no position)

This model is trained with the **Differentiable Sharpe Ratio** loss
(Section 4.2) rather than MSE, directly optimising risk-adjusted returns.

**Default configuration:** $d_{\text{model}} = 64$, $n_{\text{heads}} = 4$, $n_{\text{layers}} = 2$, $d_{ff} = 256$, dropout $= 0.1$

---

## 4. Loss Functions

**Source:** `quant/models/losses.py`

### 4.1 Combined Loss (MSE + Directional)

The primary training objective for return-prediction models:

$$\mathcal{L} = w_{\text{mse}} \cdot \mathcal{L}_{\text{MSE}} + w_{\text{dir}} \cdot \mathcal{L}_{\text{dir}}$$

#### MSE Component

$$\mathcal{L}_{\text{MSE}} = \frac{1}{BH}\sum_{b=1}^{B}\sum_{h=1}^{H} (\hat{y}_{b,h} - y_{b,h})^2$$

#### Directional Loss Component

Penalises predictions with incorrect sign relative to the actual return,
with the penalty proportional to the magnitude of the actual return:

$$\mathcal{L}_{\text{dir}} = \frac{w_p}{BH}\sum_{b,h} \mathbb{1}\!\left[\text{sgn}(\hat{y}_{b,h}) \neq \text{sgn}(y_{b,h})\right] \cdot |y_{b,h}|$$

This asymmetrically penalises directional misses more heavily when the
actual move is large, encouraging the model to get the sign right on
significant price moves.

**Default weights:** $w_{\text{mse}} = 1.0$, $w_{\text{dir}} = 0.5$, $w_p = 1.0$

### 4.2 Differentiable Sharpe Ratio

Used to train the MomentumTransformer (Section 3.4).  The loss directly
optimises the (negative) Sharpe ratio of the strategy's returns:

Given model positions $p_t \in [-1, +1]$ and realised asset returns $r_t$:

$$R_t^{\text{port}} = p_t \cdot r_t$$

$$\mathcal{L}_{\text{Sharpe}} = -\sqrt{A} \cdot \frac{\bar{R}^{\text{port}}}{\text{Std}(R^{\text{port}}) + \epsilon}$$

where $\bar{R}^{\text{port}}$ and $\text{Std}(R^{\text{port}})$ are computed
over the time dimension, and $A = 252$ is the annualisation factor.  The
negative sign ensures that minimising the loss maximises the Sharpe ratio.

Computation is batched:

$$\mathcal{L} = -\frac{1}{B}\sum_{b=1}^{B} \sqrt{A}\;\frac{\text{mean}_t(p_{b,t} \cdot r_{b,t})}{\text{std}_t(p_{b,t} \cdot r_{b,t}) + \epsilon}$$

---

## 5. PPO for Portfolio Optimisation

**Source:** `quant/models/rl_portfolio.py`

A Proximal Policy Optimisation (PPO) actor-critic agent that learns portfolio
weight allocations over multiple assets.

### 5.1 Network Architecture

```
State [state_dim] --> SharedFeatureNet --> [hidden]
                                              |
                        +---------+-----------+
                        |                     |
                   ActorHead              CriticHead
                   [n_assets]                [1]
                        |                     |
               Normal(mean, exp(log_std))   V(s)
                        |
                    rsample()
                        |
                   softmax() --> Portfolio weights [n_assets]
```

**Shared Feature Network:**

$$\mathbf{f}(s) = \text{GELU}(W_2\;\text{GELU}(W_1 s + b_1) + b_2)$$

**Actor (Policy):**

$$\boldsymbol{\mu}(s) = \text{MLP}_{\text{actor}}(\mathbf{f}(s))$$

$$\boldsymbol{\sigma} = \exp(\boldsymbol{\theta}_{\log\sigma}) \quad \text{(learnable per-asset)}$$

$$\mathbf{a}^{\text{raw}} \sim \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$$

$$\mathbf{w} = \text{softmax}(\mathbf{a}^{\text{raw}}) \quad \text{(portfolio weights, sum to 1)}$$

**Critic (Value):**

$$V(s) = \text{MLP}_{\text{critic}}(\mathbf{f}(s))$$

### 5.2 PPO Objective

The clipped surrogate objective:

$$\rho_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

$$\mathcal{L}^{\text{CLIP}} = -\mathbb{E}_t\!\left[\min\!\left(\rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\,\hat{A}_t\right)\right]$$

where $\varepsilon$ is the clipping parameter (default 0.2) and $\hat{A}_t$
is the generalised advantage estimate.

#### Generalised Advantage Estimation (GAE)

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

where $\gamma = 0.99$ (discount) and $\lambda = 0.95$ (GAE parameter).

#### Full PPO Loss

$$\mathcal{L} = \mathcal{L}^{\text{CLIP}} + c_v \cdot \mathcal{L}^{\text{value}} - c_e \cdot \mathcal{H}[\pi_\theta]$$

| Component | Formula | Weight |
|-----------|---------|--------|
| Policy loss | $\mathcal{L}^{\text{CLIP}}$ | 1.0 |
| Value loss | $\frac{1}{2}\mathbb{E}[(V_\theta(s_t) - V_t^{\text{target}})^2]$ | $c_v = 0.5$ |
| Entropy bonus | $\mathcal{H}[\pi] = -\mathbb{E}[\log\pi_\theta(a|s)]$ | $c_e = 0.01$ |

**Default configuration:** hidden $= 128$, layers $= 2$, $\varepsilon = 0.2$, $\gamma = 0.99$, $\lambda = 0.95$

---

## 6. Walk-Forward Cross-Validation

**Source:** `quant/data/dataset.py` -- `WalkForwardSplitter`

Walk-forward validation uses expanding training windows with a temporal gap
to prevent data leakage from the future.

### 6.1 Splitting Algorithm

Given $N$ total samples and $K$ desired splits:

$$\text{test\_size} = \max\!\left(1,\; \left\lfloor\frac{N \cdot r_{\text{test}}}{K}\right\rfloor\right)$$

For split $i \in \{0, 1, \ldots, K-1\}$:

$$\text{test\_end}_i = N - (K - 1 - i) \cdot \text{test\_size}$$

$$\text{test\_start}_i = \text{test\_end}_i - \text{test\_size}$$

$$\text{train\_end}_i = \text{test\_start}_i - G$$

$$\text{train}_i = [0,\; \text{train\_end}_i), \quad \text{test}_i = [\text{test\_start}_i,\; \text{test\_end}_i)$$

where $G$ is the gap size (default 10 bars) and $r_{\text{test}}$ is the test
ratio (default 0.2).

### 6.2 Visual Representation

```
Split 1:  [====TRAIN====]---gap---[TEST]
Split 2:  [=======TRAIN=======]---gap---[TEST]
Split 3:  [===========TRAIN===========]---gap---[TEST]
Split 4:  [===============TRAIN===============]---gap---[TEST]
Split 5:  [===================TRAIN===================]---gap---[TEST]
```

The final model is trained on the last (largest) split.

### 6.3 Guarantee

For each split $(S_{\text{train}}, S_{\text{test}})$:

$$\max(S_{\text{train}}) + G < \min(S_{\text{test}})$$

This ensures no temporal leakage.

**Default configuration:** $K = 5$ splits, $r_{\text{test}} = 0.2$, $G = 10$ bars

### 6.4 Sliding-Window Dataset

**Source:** `quant/data/dataset.py` -- `TimeSeriesDataset`

Each sample is a sliding window:

$$\text{Input: } \mathbf{X}_i = [\mathbf{x}_i, \mathbf{x}_{i+1}, \ldots, \mathbf{x}_{i+T-1}] \in \mathbb{R}^{T \times N}$$

$$\text{Target: } \mathbf{y}_i = [y_{i+T}, y_{i+T+1}, \ldots, y_{i+T+H-1}] \in \mathbb{R}^{H}$$

where $T$ is the sequence length (default 60) and $H$ is the forecast horizon
(default 5).  Total valid samples:

$$n_{\text{samples}} = \text{len}(\text{features}) - T - H + 1$$

---

## 7. Position Sizing

**Source:** `quant/portfolio/position.py` -- `PositionSizer`

All sizing methods produce a signed quantity subject to a hard cap:

$$|\text{qty} \cdot P| \leq C \cdot \text{max\_position\_pct}$$

where $P$ is the current price, $C$ is available capital, and
$\text{max\_position\_pct} = 0.10$ (10% of capital per position).

### 7.1 Kelly Criterion

Optimal bet sizing based on edge and odds:

$$f^* = \frac{p \cdot b - q}{b}$$

where:
- $p$ = win probability (from signal confidence, clipped to $[0.01, 0.99]$)
- $q = 1 - p$
- $b$ = average win/loss ratio (from signal metadata, default 1.5)

A **fractional Kelly** is applied for safety:

$$f_{\text{frac}} = f^* \cdot k, \quad k = 0.25 \text{ (quarter-Kelly)}$$

The final quantity:

$$\text{qty} = \frac{C \cdot f_{\text{frac}}}{P}$$

Negative $f^*$ values are floored at zero (no bet when edge is negative).

### 7.2 Volatility-Adjusted (ATR-Based)

Risk a confidence-scaled fraction of capital per ATR unit:

$$\text{risk\_frac} = f_{\text{base}} \cdot \text{confidence}$$

$$\text{stop\_distance} = \text{ATR} \cdot m_{\text{sl}}$$

$$\text{qty} = \frac{C \cdot \text{risk\_frac}}{\text{stop\_distance}}$$

where:
- $f_{\text{base}} = 0.02$ (2% risk per trade)
- $\text{confidence} \in [0, 1]$ from the trading signal
- $m_{\text{sl}} = 2.0$ (stop-loss ATR multiplier)

If ATR is unavailable (zero), this falls back to fixed-fraction sizing.

### 7.3 Fixed Fraction

Simple allocation proportional to confidence:

$$\text{qty} = \frac{C \cdot f_{\text{base}} \cdot \text{confidence}}{P}$$

where $f_{\text{base}} = 0.02$.

---

## 8. Portfolio Optimisation

**Source:** `quant/portfolio/optimizer.py`

All methods solve for weights $\mathbf{w} \in \mathbb{R}^n$ subject to:

$$\sum_{i=1}^{n} w_i = 1, \quad w_i \geq 0 \;\;\forall i$$

(fully invested, long-only).  Optimisation uses **scipy SLSQP** with
$\text{ftol} = 10^{-12}$.

### 8.1 Mean-Variance

Maximise the risk-adjusted utility:

$$\max_{\mathbf{w}} \;\; \mathbf{w}^\top \boldsymbol{\mu} - \frac{\gamma}{2}\,\mathbf{w}^\top \Sigma\, \mathbf{w}$$

equivalently minimise:

$$\min_{\mathbf{w}} \;\; -\mathbf{w}^\top \boldsymbol{\mu} + \frac{\gamma}{2}\,\mathbf{w}^\top \Sigma\, \mathbf{w}$$

where:
- $\boldsymbol{\mu}$ = expected returns (provided or estimated from historical mean)
- $\Sigma$ = sample covariance matrix of returns
- $\gamma$ = risk aversion parameter (default 1.0)

### 8.2 Minimum Variance

Ignore expected returns entirely and minimise portfolio variance:

$$\min_{\mathbf{w}} \;\; \mathbf{w}^\top \Sigma\, \mathbf{w}$$

### 8.3 Risk Parity

Equalise the marginal risk contribution of each asset.  The risk contribution
of asset $i$ is:

$$\text{RC}_i = \frac{w_i \cdot (\Sigma \mathbf{w})_i}{\sigma_p}$$

where $\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}$ is portfolio volatility.

The objective minimises the squared deviation from equal risk:

$$\min_{\mathbf{w}} \;\; \sum_{i=1}^{n} \left(\frac{\text{RC}_i}{\sum_j \text{RC}_j} - \frac{1}{n}\right)^2$$

### 8.4 Equal Weight

No optimisation required:

$$w_i = \frac{1}{n} \;\;\forall i$$

### 8.5 Post-Processing

After optimisation, weights are clipped to $[0, \infty)$ (removing numerical
noise) and renormalised to sum to 1.  If all weights collapse to near-zero,
the system falls back to equal weighting.

---

## 9. Risk Management

**Source:** `quant/portfolio/risk.py` -- `RiskManager`

The risk manager runs four portfolio-level checks and two position-level checks.
All checks must pass for a trade to proceed.

### 9.1 Value at Risk (VaR 95%)

Historical VaR is estimated as the 5th percentile of the returns distribution:

$$\text{VaR}_{95} = -\text{Percentile}(\{r_1, r_2, \ldots, r_T\}, 5)$$

The result is expressed as a positive number (loss).  A minimum of 20
observations is required for reliable estimation.

**Check:** Trading is halted if $\text{VaR}_{95} > 0.05$ (5%).

### 9.2 Maximum Drawdown

$$\text{RunMax}_t = \max_{i \leq t} E_i$$

$$\text{DD}_t = \frac{\text{RunMax}_t - E_t}{\text{RunMax}_t}$$

$$\text{MaxDD} = \max_t \text{DD}_t$$

where $E_t$ is the equity at time $t$.

**Check:** Trading is halted if $\text{MaxDD} > 0.15$ (15%).

### 9.3 Daily Loss Limit

$$\text{DailyLoss\%} = \frac{-\text{PnL}_{\text{today}}}{E_{\text{current}}}$$

**Check:** Trading is halted if $\text{DailyLoss\%} > 0.03$ (3%).

### 9.4 Position Concentration Limit

For each open position:

$$\text{Concentration}_i = \frac{|q_i \cdot P_i^{\text{current}}|}{E_{\text{total}}}$$

**Check:** Violated if $\text{Concentration}_i > 0.10$ (10%) for any position.

### 9.5 ATR-Based Stop-Loss and Take-Profit

**Stop-loss:**

$$\text{Stop}_{\text{long}} = P_{\text{entry}} - m_{\text{sl}} \cdot \text{ATR}$$

$$\text{Stop}_{\text{short}} = P_{\text{entry}} + m_{\text{sl}} \cdot \text{ATR}$$

**Take-profit:**

$$\text{TP}_{\text{long}} = P_{\text{entry}} + m_{\text{tp}} \cdot \text{ATR}$$

$$\text{TP}_{\text{short}} = P_{\text{entry}} - m_{\text{tp}} \cdot \text{ATR}$$

**Default multipliers:** $m_{\text{sl}} = 2.0$, $m_{\text{tp}} = 3.0$

The risk/reward ratio is therefore $3:2 = 1.5:1$ by default.

---

## 10. Ensemble Methods

**Source:** `quant/strategies/ensemble.py`

### 10.1 Sharpe-Weighted Voting

The ensemble maintains per-model weights $\{w_i\}_{i=1}^{M}$ that are
updated online based on each model's rolling Sharpe ratio:

**Weight update:**

$$\tilde{w}_i = \max(\text{Sharpe}_i, 0) + \epsilon, \quad \epsilon = 10^{-6}$$

$$w_i = \max\!\left(\frac{\tilde{w}_i}{\sum_j \tilde{w}_j},\; w_{\min}\right)$$

then renormalise so $\sum_i w_i = 1$.

The minimum weight floor $w_{\min} = 0.05$ prevents any model from being
entirely silenced, preserving ensemble diversity.

**Initial weights:** All models start with equal weight $w_i = 1/M$.

### 10.2 Confidence Aggregation

For each symbol, the ensemble aggregates signals from all sub-strategies:

**Step 1 -- Weighted direction vote:**

$$V = \frac{\sum_{i=1}^{M} w_i \cdot d_i}{\sum_{i=1}^{M} w_i}$$

where $d_i \in \{-1, 0, +1\}$ is the direction from model $i$.

**Step 2 -- Consensus direction:**

$$D = \begin{cases} +1 & \text{if } V > 0.1 \\ -1 & \text{if } V < -0.1 \\ 0 & \text{otherwise (dead zone)} \end{cases}$$

**Step 3 -- Weighted confidence (agreeing models only):**

$$c = \frac{\sum_{i: d_i = D} w_i \cdot c_i}{\sum_{i: d_i = D} w_i}$$

where $c_i$ is the confidence from model $i$.  The confidence is clipped to
$[0, 1]$.

**Step 4 -- Target position:**

$$p = \text{clip}(D \cdot c, -1, 1)$$

**Minimum confidence gate:** Signals with $c < 0.6$ are suppressed (not
emitted).

---

## 11. Performance Metrics

**Source:** `quant/backtest/metrics.py`

The system computes 17 metrics from the equity curve and trade log.

### 11.1 Return Metrics

$$\text{Total Return} = \frac{E_T - E_0}{E_0}$$

$$\text{Annualised Return} = \left(\frac{E_T}{E_0}\right)^{252/n} - 1$$

where $n$ is the number of bars.

### 11.2 Risk-Adjusted Ratios

**Sharpe Ratio** (assuming zero risk-free rate):

$$\text{SR} = \frac{\bar{r}}{\sigma_r} \cdot \sqrt{252}$$

where $\bar{r}$ and $\sigma_r$ are the mean and standard deviation (with
Bessel's correction, $\text{ddof}=1$) of per-bar returns.

**Sortino Ratio** (downside deviation):

$$\text{Sortino} = \frac{\bar{r}}{\sigma_d} \cdot \sqrt{252}$$

$$\sigma_d = \sqrt{\frac{1}{n_d}\sum_{r_t < 0} r_t^2}$$

where $n_d$ is the count of negative returns.

**Calmar Ratio:**

$$\text{Calmar} = \frac{\text{Annualised Return}}{\text{Max Drawdown}}$$

### 11.3 Drawdown Metrics

$$\text{Max Drawdown} = \max_t \frac{\max_{i \leq t} E_i - E_t}{\max_{i \leq t} E_i}$$

$$\text{Max DD Duration} = \max\!\bigl(\text{consecutive bars below running peak}\bigr)$$

### 11.4 Trade-Level Metrics

| Metric | Formula |
|--------|---------|
| Win Rate | $n_{\text{wins}} / n_{\text{trades}}$ |
| Profit Factor | $\sum \text{wins} / |\sum \text{losses}|$ |
| Average Win | $\bar{w} = \text{mean}(\text{PnL} \mid \text{PnL} > 0)$ |
| Average Loss | $\bar{l} = \text{mean}(\text{PnL} \mid \text{PnL} < 0)$ |
| Expectancy | $\text{mean}(\text{PnL})$ per trade |
| Avg Trade Duration | Mean of exit_time $-$ entry_time |

---

## Appendix A: Configuration Defaults Summary

| Parameter | Value | Module |
|-----------|-------|--------|
| `seq_len` | 60 bars | ModelConfig |
| `forecast_horizon` | 5 bars | ModelConfig / FeatureConfig |
| `rolling_window` | 252 | FeatureConfig |
| `correlation_threshold` | 0.95 | FeatureConfig |
| `bocpd_lambda` | 200 | FeatureConfig |
| `learning_rate` | 1e-3 | TrainingConfig |
| `weight_decay` | 1e-5 | TrainingConfig |
| `max_grad_norm` | 1.0 | TrainingConfig |
| `early_stopping_patience` | 10 epochs | TrainingConfig |
| `n_splits` | 5 | TrainingConfig |
| `gap_size` | 10 bars | TrainingConfig |
| `max_position_pct` | 10% | TradingConfig |
| `kelly_fraction` | 0.25 | TradingConfig |
| `max_drawdown` | 15% | TradingConfig |
| `daily_loss_limit` | 3% | TradingConfig |
| `max_var_95` | 5% | TradingConfig |
| `min_confidence` | 0.6 | TradingConfig |
| `stop_loss_atr_mult` | 2.0x | TradingConfig |
| `take_profit_atr_mult` | 3.0x | TradingConfig |
| `slippage_bps` | 5 bp | TradingConfig |
| `commission_bps` | 10 bp | TradingConfig |
| `initial_capital` | $100,000 | BacktestConfig |

## Appendix B: References

1. Adams, R.P. & MacKay, D.J.C. (2007). *Bayesian Online Changepoint Detection.* arXiv:0710.3742.
2. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
3. Liu, Y. et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.* ICLR 2024.
4. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
5. Kelly, J.L. (1956). *A New Interpretation of Information Rate.* Bell System Technical Journal.
6. Markowitz, H. (1952). *Portfolio Selection.* The Journal of Finance.
7. Maillard, S., Roncalli, T., & Teiletche, J. (2010). *The Properties of Equally Weighted Risk Contribution Portfolios.* Journal of Portfolio Management.
