## Normalized Inverse Gaussian Process

type NIGP   <:  DistributionOnDistributions
    beta          ::  Float64
    base          ::  Array{Distribution}
    recursive     ::  Bool
    NIGP(beta::Float64, base, recursive::Bool=false) = begin
        new(beta, isa(base, Array) ? base : [base], recursive)
    end
end

function Distributions.rand(d::NIGP)
    return  d.recursive ? NIGPRecsample(d.beta, d.base) : NIGPsample(d.beta, d.base)
end

type NIGPsample <: NormalizedRandomMeasure
    beta          ::  Float64
    base          ::  Array{Distribution}
    atoms         ::  Vector{Any}
    weights       ::  Vector{Float64}
    T_surplus     ::  Float64
    NIGPsample(beta::Float64, base) = begin
        new(beta, base, [], Array(Float64,0), 1)
    end
end

type NIGPRecsample <: NormalizedRandomMeasureRec
    beta          ::  Float64
    base          ::  Array{Distribution}
    atoms         ::  Dict{Int, Any}
    sticks        ::  Array{Float64}
    T_surplus     ::  Float64
    NIGPRecsample(beta::Float64, base) = begin
        new(beta, base, Dict{Int, Any}(), Array(Float64,0), 1)
    end
end

function sampleStick(d::NIGPsample)
    Z = rand(Levy(0, 1)) # ~ St_1/2(1)
    X = rand(GeneralizedInverseGaussian(d.beta/d.T_surplus, 1, -0.5*(length(d.weights)+1)))
    return X / (X + Z)
end

function sampleStick(d::NIGPRecsample)
    Z = rand(Levy(0, 1)) # ~ St_1/2(1)
    X = rand(GeneralizedInverseGaussian(d.beta/d.T_surplus, 1, -0.5*(length(d.sticks)+1)))
    V = X / (X + Z)
    d.T_surplus *= (1 - V)
    return V
end
