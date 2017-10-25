type PYP           <:  DistributionOnDistributions
    alpha         ::  Float64
    theta         ::  Float64
    base          ::  Distribution
    recursive     ::  Bool
    PYP(alpha::Float64, theta::Float64, base, recursive::Bool=false) = begin
        new(alpha, theta, base, recursive)
    end
end

function Distributions.rand(d::PYP)
    return d.recursive ? PYPRecsample(d.alpha, d.theta, d.base) : PYPsample(d.alpha, d.theta, d.base)
end

type PYPsample <: NormalizedRandomMeasure
    alpha         ::  Float64
    theta         ::  Float64
    base          ::  Distribution
    atoms         ::  Vector{Any}
    weights       ::  Vector{Float64}
    T_surplus     ::  Float64
    PYPsample(alpha::Float64, theta::Float64, base::Distribution) = begin
        new(alpha, theta, base, [], Array(Float64,0), 1)
    end
end

type PYPRecsample <: NormalizedRandomMeasureRec
    alpha         ::  Float64
    theta         ::  Float64
    base          ::  Distribution
    atoms         ::  Dict{Int, Any}
    sticks        ::  Array{Float64}
    PYPRecsample(alpha::Float64, theta::Float64, base::Distribution) = begin
        new(alpha, theta, base, Dict{Int, Any}(), Array(Float64,0))
    end
end

function sampleStick(d::PYPsample)
    return rand(Beta(d.alpha, d.theta + (length(d.weights) - 1) * d.alpha))
end

function sampleStick(d::PYPRecsample)
    return rand(Beta(d.alpha, d.theta + (length(d.sticks) - 1) * d.alpha))
end
