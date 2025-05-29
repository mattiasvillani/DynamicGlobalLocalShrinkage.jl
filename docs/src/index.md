```@meta
CurrentModule = DynamicGlobalLocalShrinkage
```

# DynamicGlobalLocalShrinkage

Documentation for dynamic global-local shrinkage process priors for state-space models  [DynamicGlobalLocalShrinkage](https://github.com/mattiasvillani/DynamicGlobalLocalShrinkage.jl).

## Installation
Install from the Julia package manager (via Github) by typing `]` in the Julia REPL:
```
] add git@github.com:mattiasvillani/DynamicGlobalLocalShrinkage.jl.git
```

## Features

The package implements a Gibbs sampler for dynamic global-local shrinkage process priors which can be used in state-space models. The algorithm is essentially the one presented in Kowal et al. (2019)[^1] but some minor changes due to a somewhat different prior for the hyperparameters. The sampler in Kowal et al. (2019) builds on the work of Kim et al. (1998)[^2] and Kastner and Fruhwirth-Schnatter (2014)[^3] for stochastic volatility models, and Rue (2001)[^4] for fast sampling of high-dimensional Gaussian distributions with sparse precision matrix.

## References

[^1]: Kowal, D. R., Matteson, D. S., and Ruppert, D. (2019). Dynamic shrinkage processes. *Journal of the Royal Statistical Society: Series B* (Statistical Methodology)*, 81(4):781–804.
[^2]: Kim, S., Shephard, N., and Chib, S. (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. *The Review of Economic Studies*, 65(3):361–393.
[^3]: Kastner, G. and Fruhwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. *Computational Statistics & Data Analysis*, 76:408–423.
[^4]: Rue, H. (2001). Fast sampling of Gaussian Markov random fields. *Journal of the Royal Statistical Society: Series B*, 63(2):325–338.
