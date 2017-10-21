type DP           <:  DistributionOnDistributions
    alpha         ::  Real
    base          ::  Array{Distribution}
    recursive     ::  Bool
    DP(alpha::Float64, base, recursive::Bool=false) = begin
        new(alpha, isa(base, Array) ? base : [base], recursive)
    end
end

function Distributions.rand(d::DP)
    return d.recursive ? DPRecsample(d.alpha, d.base) : DPsample(d.alpha, d.base)
end

type DPsample <: NormalizedRandomMeasure
    alpha         ::  Float64
    base          ::  Array{Distribution}
    atoms         ::  Vector{Any}
    weights       ::  Vector{Float64}
    T_surplus     ::  Float64
    DPsample(alpha::Float64, base::Array{Distribution}) = begin
        new(alpha, base, [], Array(Float64,0), 1)
    end
end

type DPRecsample <: NormalizedRandomMeasureRec
    alpha         ::  Float64
    base          ::  Array{Distribution}
    atoms         ::  Dict{Int, Any}
    sticks        ::  Array{Float64}
    DPRecsample(alpha::Float64, base::Array{Distribution}) = begin
        new(alpha, base, Dict{Int, Any}(), Array(Float64,0))
    end
end

function sampleStick(d::Union{DPsample,DPRecsample})
    return rand(Beta(1, d.alpha))
end
