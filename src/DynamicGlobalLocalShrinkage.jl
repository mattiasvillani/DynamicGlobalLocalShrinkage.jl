module DynamicGlobalLocalShrinkage

using Distributions, LinearAlgebra, PDMats, Statistics
using BandedMatrices, SparseArrays, PolyaGammaSamplers
using StatsBase

using Colors
colors = Base.parse.(Colorant,["#6C8EBF", "#c0a34d", "#780000", "#007878",     
"#bf9d6c", "#3A6B35", "#b5c6df","#eadaaa", 
"#bb989a", "#98bbb9", "#bf8d6c", "#CBD18F"])
export colors

include("GibbsDSPcomponents.jl")
export Updateξ, Updateϕ, Updateμ, UpdateμNC, Updateσ²ₙ
export Update_h, UpdateMixAlloc
export SetUpLogChi2Mixture, ScaledInverseChiSq

include("GibbsDSP.jl")
export update_dsp, update_dsp!

include("Utils.jl")
export quantile_multidim, setOffset!

end
