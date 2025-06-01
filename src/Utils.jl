
""" 
    quantile(A, p; dims, kwargs...)

Compute the quantiles `p` of a multidimensional array `A` along specified dimensions `dims`.
""" 
quantile_multidim(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims)

# Set the offset
function setOffset!(offset, ν, offSetMethod)
    if offSetMethod == "kowal"
        for j = 1:size(ν,2) 
            offset[:,j] .= eps() + any(ν[:,j].^2 .< 10^-16) * 
                maximum([10^-8, mad(ν[:,j])/10^6])
        end
    end
end
