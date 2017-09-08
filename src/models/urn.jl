using Memoize
import Turing.vectorize
import Base.rand, Base.getindex

function pickStick(sticks, J)
    if Bool(rand(Bernoulli(sticks(J))))
        return J
    else
        return pickStick(sticks, J+1)
    end
end

function makeSticks(alpha)
    @memoize function sticks(index)
      return rand(Beta(1, alpha))
    end
    return () -> pickStick(sticks, 1)
end

type PolyaUrn <: DiscreteUnivariateDistribution
    alpha :: Float64
    makeSticks :: Function
    PolyaUrn(alpha::Float64) = new(alpha, makeSticks(alpha))
end

function rand(urn::PolyaUrn)
    res = urn.makeSticks()
    return res
end

immutable IArray <: ContinuousMultivariateDistribution
    base
end

type IArraySample
    base
    dict
    IArraySample(base) = new(base, Dict{Int64, Float64}())
end

function getindex(arr::IArraySample, i::Int64)
    if !haskey(arr.dict, i)
        arr.dict[i] = rand(arr.base)
    end
    return arr.dict[i]
end

# function rand(arr::IArraySample, i::Int64)
#     if !haskey(arr.dict, i)
#         arr.dict[i] = rand(arr.base)
#     end
#     return arr.dict[i]
# end

function rand(d::IArray)
    return IArraySample(d.base)
end

Distributions.logpdf(d::PolyaUrn, x::Int64) = zero(x)
# Distributions.logpdf{T<:Real}(d::PolyaUrn, x::T) = zero(x)
@inline vectorize(d::IArray, r::IArraySample) = Vector{Real}(collect(values(r.dict)))

using Turing
data = [9172.0;9350.0;9483.0;9558.0;9775.0;10227.0;10406.0;16084.0;16170.0;18419.0;18552.0;18600.0;18927.0;19052.0;19070.0;19330.0;19343.0;19349.0;19440.0;19473.0;19529.0;19541.0;19547.0;19663.0;19846.0;19856.0;19863.0;19914.0;19918.0;19973.0;19989.0;20166.0;20175.0;20179.0;20196.0;20215.0;20221.0;20415.0;20629.0;20795.0;20821.0;20846.0;20875.0;20986.0;21137.0;21492.0;21701.0;21814.0;21921.0;21960.0;22185.0;22209.0;22242.0;22249.0;22314.0;22374.0;22495.0;22746.0;22747.0;22888.0;22914.0;23206.0;23241.0;23263.0;23484.0;23538.0;23542.0;23666.0;23706.0;23711.0;24129.0;24285.0;24289.0;24366.0;24717.0;24990.0;25633.0;26960.0;26995.0;32065.0;32789.0;34279.0]
data /= 1e4
meanMean = 2.17; meanPrecision = 0.63; precisionShape = 2.0; precisionInvScale = 0.2/6.34

@model infiniteMixture(y) = begin
  N = length(y)
  m ~ Normal(meanMean, 1.0/sqrt(meanPrecision))
  s ~ Gamma(precisionShape, 1.0/precisionInvScale)
  x = rand(IArray(Normal(m, 1.0/sqrt(s)))) # Infinite Array
  urn = PolyaUrn(1.72)

  x, z = tzeros(N), tzeros(Int, N)
  for i in 1:N
    z[i] ~ urn
    x[i] = x[z[i]]
    # x[i] ~ x(z[i])
    y[i] ~ Normal(x[i], 1.0/sqrt(s))
  end
end

# sampler = Gibbs(3, CSMC(50, 1, :z), HMC(1, 0.2, 3, :m, :s))
sampler = Gibbs(100, CSMC(40, 1, :z), HMC(1, 0.2, 3, :m, :s))
results = sample(infiniteMixture(data), sampler)

mixtureComponentsRes = results[:z]
mixtureComponents = zeros(length(data), length(mixtureComponentsRes))
for j in 1:size(mixtureComponents,2)
    mixtureComponents[:,j] = mixtureComponentsRes[j]
end
include("plot.jl")
linescatter(mixtureComponents[1,:],mixtureComponents[end,:])
linescatter(results[:m])
plot_histogram(results[:m])
