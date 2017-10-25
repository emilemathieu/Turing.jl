## Normalized Generalized Gamma Process
# /!\ with sigma=1/2 for now

type NGGP   <:  DistributionOnDistributions
    sigma         ::  Float64
    b             ::  Float64
    base          ::  Array{Distribution}
    recursive     ::  Bool
    NGGP(sigma::Float64, b::Float64, base, recursive::Bool=false) = begin
        if sigma != 0.5 error("only handle sigma=1/2 for now") end
        new(0.5, b, isa(base, Array) ? base : [base], recursive)
    end
end

function Distributions.rand(d::NGGP)
    return  d.recursive ? NGGPRecsample(d.sigma, d.b, d.base) : NGGPsample(d.sigma, d.b, d.base)
end

type NGGPsample <: PoissonKingmanMeasure
    sigma         ::  Float64
    b             ::  Float64
    base          ::  Array{Distribution}
    atoms         ::  Vector{Any}
    weights       ::  Vector{Float64}
    T             ::  Float64
    T_surplus     ::  Float64
    NGGPsample(sigma::Float64, b::Float64, base) = begin
        T = rand(ExpTiltedSigma(sigma, b))
        new(sigma, b, base, [], Array(Float64,0), T, T)
    end
end

type NGGPRecsample <: PoissonKingmanMeasureRec
    sigma         ::  Float64
    b             ::  Float64
    base          ::  Array{Distribution}
    atoms         ::  Dict{Int, Any}
    sticks        ::  Array{Float64}
    T             ::  Float64
    T_surplus     ::  Float64
    NGGPRecsample(sigma::Float64, b::Float64, base) = begin
        T = rand(ExpTiltedSigma(sigma, b))
        new(sigma, b, base, Dict{Int, Any}(), Array(Float64,0), T, T)
    end
end

function sampleWeight(d::Union{NGGPsample,NGGPRecsample})
    X = sqrt(rand(Gamma(3/4, 1)))
    # Y = sqrt(rand(InverseGamma(1/4, 1/(4^3 * d.T^2 * d.Vcomp2))))
    Y = sqrt(rand(InverseGamma(1/4, 1/(64 * d.T_surplus^2))))
    return d.T_surplus * X / (X + Y)
end
