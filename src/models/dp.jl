type DP           <:  DistributionOnDistributions
    alpha         ::  Real
    base          ::  Distribution
end

function Distributions.rand(d::DP)
    return DPsample(d.alpha, d.base)
end

type DPsample <: NormalizedRandomMeasure
    alpha         ::  Float64
    base          ::  Distribution
    atoms         ::  Vector{Float64}
    lengths       ::  Vector{Float64}
    T_surplus     ::  Float64
    DPsample(alpha::Float64, base) = begin
        new(alpha, base, Array(Float64,0), Array(Float64,0), 1)
    end
end

function sampleStick(d::DPsample)
    return rand(Beta(1, d.alpha))
end
