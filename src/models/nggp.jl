## Normalized Generalized Gamma Process
# /!\ with sigma=1/2

type NGGP   <:  DistributionOnDistributions
    sigma         ::  Float64
    b             ::  Float64
    base          ::  Array{Distribution}
    recursive     ::  Bool
    NGGP(sigma::Float64, b::Float64, base, recursive::Bool=false) = begin
        if sigma != 0.5 error("only handle sigma=1/2 for now") end
        if recursive error("No recursive yet") end
        new(0.5, b, isa(base, Array) ? base : [base], recursive)
    end
end

function Distributions.rand(d::NGGP)
    # return  d.recursive ? NGGPRecsample(d.b, d.base) : NGGPsample(d.b, d.base)
    return  NGGPsample(d.sigma, d.b, d.base)
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

function sampleWeight(d::NGGPsample)
    X = sqrt(rand(Gamma(3/4, 1)))
    # Y = sqrt(rand(InverseGamma(1/4, 1/(64 * d.T^2 * d.Vcomp2))))
    Y = sqrt(rand(InverseGamma(1/4, 1/(4^3 * d.T_surplus^2))))
    # V = X / (X + Y)
    # d.Vcomp2 *= (1 - V)^2
    # return V * d.T_surplus
    return d.T_surplus * X / (X + Y)
end
