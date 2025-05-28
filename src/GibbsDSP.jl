""" 
    ScaledInverseChiSq(ν,τ²) 

Scaled inverse chi-squared distribution with parameters `ν` and `τ²`.
""" 
ScaledInverseChiSq(ν,τ²) = InverseGamma(ν/2,ν*τ²/2)


""" 
    Updateξ(y, ϕ, σ²ₙ, μ, α, β) 

Sampling update step for the Polya-Gamma variables
""" 
function Updateξ(y, ϕ, σ²ₙ, μ, α, β)
    η = (y[2:end] .- μ .- ϕ .* (y[1:end-1] .- μ)) ./ sqrt.(σ²ₙ)
    return rand.(PolyaGammaPSWSampler.(Int(α+β), [(y[1]-μ)/sqrt.(σ²ₙ);η])) # ξ₀,ξ₁,...ξ\_T
end


""" 
    Updateϕ(y, ξ, μ, σ²ₙ, ϕ₀, κ₀) 

Sampling update the AR coefficient in the log-volatility evolution
""" 
function Updateϕ(y, ξ, μ, σ²ₙ, ϕ₀, κ₀)
    zt = sqrt.(ξ[2:end]) .* (y[2:end] .- μ)
    zt_1 = sqrt.(ξ[2:end]) .* (y[1:end-1] .- μ)
    xx = zt_1'*zt_1
    xy = zt_1'*zt
    κₜ² = 1/(xx/σ²ₙ + 1/κ₀^2)
    w = (xx/σ²ₙ)*κₜ² 
    ϕ̂ = xy/xx
    ϕₜ = w*ϕ̂  + (1-w)*ϕ₀
    return rand(Truncated(Normal(ϕₜ, sqrt(κₜ²)), -1, 1))
end


""" 
    Updateμ(y, ξ, ϕ, σ²ₙ, m₀, σ₀) 

Sampling update of the mean in the logvariance evolution - Centered parameterization
""" 
function Updateμ(y, ξ, ϕ, σ²ₙ, m₀, σ₀)
    ξ̃ = [ξ[1];(1-ϕ)^2*ξ[2:end]];
    σₜ² = 1/( sum(ξ̃)/σ²ₙ + 1/σ₀^2 ) 
    v = 1 - (1/σ₀^2)*σₜ²
    z = [ y[1] ; (y[2:end] .- ϕ*y[1:end-1])/(1-ϕ) ]
    μ̂ = sum(ξ̃ .* z)/sum(ξ̃)
    μₜ = v*μ̂  + (1 - v)*m₀
    return rand(Normal(μₜ, sqrt(σₜ²)))
end


""" 
    UpdateμNC(ỹ, h̃, m, v, m₀, σ₀) 

Sampling update of the mean in the logvariance evolution - Non-centered parameterization

""" 
function UpdateμNC(ỹ, h̃, m, v, m₀, σ₀)
    # ỹ = log.(ν.^2 .+ offset) for one parameter
    ytrans = ỹ - h̃ - m
    precision = 1 ./ v
    sum_precision = sum(precision)
    sum_precision_y = sum(precision .* ytrans)
    μ̂ = sum_precision_y / sum_precision
    σₜ² = 1/(sum_precision + 1/σ₀^2)
    w = sum_precision / (sum_precision + 1/σ₀^2)
    μₜ = w*μ̂  + (1 - w)*m₀
    return rand(Normal(μₜ, sqrt(σₜ²)))
end


""" 
    Updateσ²ₙ(y, ξ, ϕ, μ, ν₀, ψ₀) 

Sampling update of the variance in the log-volatility evolution
""" 
function Updateσ²ₙ(y, ξ, ϕ, μ, ν₀, ψ₀)
    T = length(y)
    η̃ = [ sqrt(ξ[1])*(y[1]-μ) ; sqrt.(ξ[2:end]) .* (y[2:end] .- μ .- ϕ*(y[1:end-1] .- μ))]
    return rand(ScaledInverseChiSq(ν₀ + T , (ν₀*ψ₀^2 + sum(η̃.^2))/(ν₀ + T) ))
end


""" 
    Update_h(ỹ, m, v, Dᵩ, ξ, ϕ, σ²ₙ, μ) 

Update log-volatility series for one parameter
""" 
function Update_h(ỹ, m, v, Dᵩ, ξ, ϕ, σ²ₙ, μ)
    T = length(ỹ)
    Dᵩ[band(-1)] .= -ϕ
    invΣₓ = Diagonal(ξ/σ²ₙ)
    invΣᵥ = Diagonal(1 ./ v)
    Qh̃ = PDSparseMat(sparse( invΣᵥ + Dᵩ' * invΣₓ * Dᵩ ))
    lh̃ = invΣᵥ*(ỹ - m .- μ)
    h̃ = rand(MvNormalCanon(lh̃, Qh̃))
    return h̃ .+ μ
end


""" 
    UpdateMixAlloc(ỹ, h, mixDist) 

Update the mixture component allocation for the log-volatility series
""" 
function UpdateMixAlloc(ỹ, h, mixDist)
    nComp = length(mixDist.components)
    T = length(ỹ)

    postDist = zeros(T,nComp)
    for (i,component) in enumerate(mixDist.components)
        postDist[:,i] = logpdf.(component, ỹ - h)
    end
    postDist = exp.(postDist .- maximum(postDist, dims = 2)) .* mixDist.prior.p' 
    postDist = postDist ./ sum(postDist, dims=2)

    return nComp .- 
        sum(repeat(rand(T), 1, nComp) .< cumsum(postDist, dims = 2), dims = 2) .+ 1
end


""" 
    UpdateInitialState(z₁, μ₀, Σ₀, A, Σₙ)

Update the initial z₀ state given the following state z₁
""" 
function UpdateInitialState(z₁, μ₀, Σ₀, A, Σₙ)
    Σ₀₁ = inv(inv(Σ₀) + PDMat(A'*(Σₙ\A)))
    μ₀₁ = Σ₀₁*(A'*(Σₙ\z₁) + inv(Σ₀)*μ₀)
    return rand(MvNormal(μ₀₁, Σ₀₁))
end


""" 
    setUpLogChi2Mixture(nComp, df) 

Sets up a Normal mixture to approximate the distribution of the log χ²₁ random variable. 

# Examples
```julia-repl
julia> histogram(log.(rand(Chisq(1),10000)), normalize = true, fillcolor = :lightgray, linecolor = :white, lw = 0.5, label = "simulated")
julia> mix = SetUpLogChi2Mixture(10, 1) # Omori et al (2007) 10-comp
julia> plot!(-10:0.01:4, pdf.(mix, -10:0.01:4), color = :black, label = "10-comp")
julia> mix = SetUpLogChi2Mixture(5, 1) # Carter-Kohn 5-component
julia> plot!(-10:0.01:4, pdf.(mix, -10:0.01:4), color = :cornflowerblue, label = "5-comp")
```
""" 
function SetUpLogChi2Mixture(nComp, df)

    if df !== 1
        error("Only df = 1 is implemented")
    end

    if nComp == 5 # Carter-Kohn 5-comp from JRSS B paper on spectral density estimation
        ω = [0.13, 0.16, 0.23, 0.22, 0.25]
        ω = ω ./ sum(ω)
        m = [-4.63, -2.87, -1.44, -.33,  0.76]  
        v = [8.75, 1.95, 0.88, 0.45, 0.41]
        return MixtureModel(Normal.(m, sqrt.(v)), ω), m, v
    end
    if nComp == 10 # Omori et al (2007) 10-comp (used in stochvol package in R)
        ω = [0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 
            0.18842, 0.12047, 0.05591, 0.01575, 0.00115]
        m = [1.92677, 1.34744, 0.73504, 0.02266, -0.85173, 
            -1.97278, -3.46788, -5.55246, -8.68384, -14.6500]
        v =  [0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 
            0.98583, 1.57469, 2.54498, 4.16591, 7.33342]
        return MixtureModel(Normal.(m, sqrt.(v)), ω), m, v
    end

    error("Only 5 or 10 components are implemented")

end


# Some small helper function used in the Gibbs sampler

# Function for converting log volatilities to Cov matrices used in the filters
LogVol2Covs(H) = PDMat.([diagm(exp.(H[t,:])) for t in 1:size(H,1)])