""" 
    S, H, ξ, ϕ, μ, σ²ₙ = update_dsp(ν, S, H, ξ, ϕ, μ, σ²ₙ) 

A single Gibbs update of all dynamic shrinkage process (DSP) parameters given a T × p matrix of parameter innovations, ν.

The columns of ν correspond to different parameters, which are assumed to be independent.
The elements in a column of ν are innovations for the log-volatility evolution, i.e.

νₜ ~ N(0, exp(hₜ/2)) for t = 1,2,...,T
hₜ = μ + ϕ(hₜ₋₁ - μ) + ηₜ, where εₜ ~ N(0,σₙ²/ξₜ)
ξₜ ~ PG(2α) is a Polya-Gamma variable.

The columns of H contains the log-volatility evolution for a given parameter: hₜ for t = 1,2,...,T. The Polya-Gamma construction gives a marginal distribution for ηₜ ~ Z(α,α,0,σₙ).

The usual square-and-log trick in stochastic volatility models is used to turn hₜ into an additive parameter 
ỹₜ = log(νₜ² + offset) = hₜ + qₜ, where qₜ ~ log χ²₁, which is approximated by a mixture of normals distribution with mixture allocation given by the T × p matrix S. The means and variances of the mixture components are given by the vectors m and v, respectively.
""" 
function update_dsp(ν, S, H, H̃, ξ, ϕ, μ, σ²ₙ, ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, 
    mixLogχ²₁, m, v, Dᵩ, offset = eps(), α = 1/2, β = 1/2, updateσₙ = false)

    p = size(ν, 2) 
    Ỹ = log.(ν.^2 .+ offset) # Ỹ = [ỹ₁, ỹ₂,..., ỹₚ] is T × p
    for k in 1:p 

        # Update mixture allocation for log χ²₁ distribution
        S[:,k] = UpdateMixAlloc(Ỹ[:,k], H̃[:,k] .+ μ[k], mixLogχ²₁)

        # Update h₁, h₂, ..., hₜ using MvNormal draw with tridiag precision matrix
        H[:,k] = Update_h(Ỹ[:,k], m[S[:,k]], v[S[:,k]], Dᵩ, ξ[:,k], ϕ[k], σ²ₙ[k], μ[k])

        # Update Polya-Gamma variables
        ξ[:,k] = Updateξ(H[:,k], ϕ[k], σ²ₙ[k], μ[k], α, β)

        # Update the AR coefficient in the logvariance evolution
        ϕ[k] = Updateϕ(H[:,k], ξ[:,k], μ[k], σ²ₙ[k], ϕ₀, κ₀)

        # Update the variance in the logvariance evolution
        if updateσₙ
            σ²ₙ[k] = Updateσ²ₙ(H[:,k], ξ[:,k], ϕ[k], μ[k], ν₀, ψ₀)
        end

        # Update the mean in the logvariance evolution - centered parameterization
        μ[k] = Updateμ(H[:,k], ξ[:,k], ϕ[k], σ²ₙ[k], m₀, σ₀)
        H̃[:,k] = H[:,k] .- μ[k] # Transform back to non-centered parameterization
    end

    return S, H, H̃, ξ, ϕ, μ, σ²ₙ
end