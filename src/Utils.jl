
""" 
    quantile(A, p; dims, kwargs...)

Compute the quantiles `p` of a multidimensional array `A` along specified dimensions `dims`.
""" 
quantile(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims)
