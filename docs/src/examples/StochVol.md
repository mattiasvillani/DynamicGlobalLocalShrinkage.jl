```@meta
EditURL = "../../../examples/StochVol/script.jl"
```

# Stochastic Volatility with Dynamic Shrinkage Process Prior

In this example we explore the joint posterior in the stochastic volatility model with a dynamic shrinkage process prior.

```math
\begin{align*}
  y_t &= \mu_y + \nu_t, \quad \nu_t \sim N(0,\exp(h_t)) \\
  h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\beta, 0, 1) \\
  x_0 &\sim N(0, \sigma^2_0)
\end{align*}
```

The above model needs to be rewritten using two tricks to make it possible to use Gibbs sampling

First, since $y_t - \mu_y = \exp(h_t/2)\tilde{\nu}_t, \, \tilde{\nu}_t \sim N(0,1)$, we can square and take logs to obtain $y^\star_t = h_t + \zeta_t \quad \zeta_t \sim \log\chi^2_1$, where $y^\star_t = \log (y_t -\mu_y)^2$ and $\log\chi^2_1$ is the distribution of the log of a chi-squared variable with 1 degree of freedom. This distribution can be approximated by a pre-determined mixture of normals distribution, $\zeta_t \overset{\mathrm{approx}}{\sim}\mathrm{MoN}(\boldsymbol{m},\boldsymbol{v}, \boldsymbol{\omega})$, where $\boldsymbol{m}$ is the vector with $K$ means, $\boldsymbol{v}$ is the vector with variances and $\boldsymbol{\omega}$ is the vector with weights for the mixture components. The 10-component mixture of normals approximation from Omori et al. (2007)[^1] is the default in the package.

Second, the Z-distribution can be expressed as mean-variance mixture of Normals. In the special case with a symmetric Z-distribution $\alpha=\beta$ with zero mean, we can express the distribution as a scale mixture of normals with a variance following a Polya-Gamma distribution

```math
\begin{equation*}
  Z(\alpha, \alpha, 0, \sigma^2_\eta) = \int N(0,\sigma^2_\eta \xi^{-1})\mathrm{PG}(\xi \vert 2\alpha,0)\, d \xi
\end{equation*}
```

The original parameters $\mu_y, h_{1:T}, \mu, \phi$ have been expanded to $\mu_y, h_{1:T}, \mu, \phi, \xi_{0:T}, I_{1:T}$, where $I_{1:T}$ is a vector with the mixture component indicators in the mixture of normals approximation to the log chi-square distribution. The complete Gibbs sampling algorithm therefore consists of the following steps:
- Sample the mixture allocation $I_{1:T}$ from a categorical distribution
- Sample log-volatility evolution $h_{0:T}$ using a sparse multivariate normal distribution
- Sample $\mu$ from a normal distribution
- Sample $\phi$ from a truncated normal distribution
- Sample the Polya-Gamma variables $\xi_{1:T}$ from a Polya-Gamma distribution
- Sample the mean $\mu_y$ from a normal distribution

First we load the required packages and set some plotting parameters.:

````julia
using Plots, Distributions, LaTeXStrings, Random
using DynamicGlobalLocalShrinkage

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

Random.seed!(123);
````

[^1]: Omori, Y., Chib, S., Shephard, N., and Nakajima, J. (2007). Stochastic volatility with leverage: Fast and efficient likelihood inference. *Journal of Econometrics*, 140(2):425-449.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

