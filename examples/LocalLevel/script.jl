# # Stochastic volatility model

# In this example we explore the joint posterior of the state $x_t$ in a local level model with a dynamic shrinkage process prior for the parameter evoluation.
#
# ```math
# \begin{align*}
#   y_t &= x_t + \epsilon_t, \quad \epsilon_t \sim N(0,1) \\
#   x_t &= x_{t-1} + \nu_t, \quad \nu_t \sim N(0,\exp(h_t)) \\
#   h_t &= \mu + \phi(h_{t-1} -\mu) + \eta_t, \quad \eta_t \sim Z(\alpha,\beta, 0, 1) \\
#   x_0 &\sim N(0, \sigma_0)  
# \end{align*}
# ```

# First some preliminaries:
using Plots, Distributions, LaTeXStrings, Random

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

Random.seed!(123);