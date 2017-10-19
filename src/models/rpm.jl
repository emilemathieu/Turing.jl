abstract DistributionOnDistributions      <: Distribution
abstract DiscreteRandomProbabilityMeasure <: DiscreteUnivariateDistribution
abstract NormalizedRandomMeasure          <: DiscreteRandomProbabilityMeasure
abstract PoissonKingmanMeasure            <: DiscreteRandomProbabilityMeasure

### DiscreteRandomProbabilityMeasure

@inline vectorize(d::DiscreteRandomProbabilityMeasure, r::Float64) = Vector{Real}([r])

Distributions.logpdf{T<:Real}(d::DiscreteRandomProbabilityMeasure, x::T) = zero(x)

function pickStick(d::DiscreteRandomProbabilityMeasure, sticks::Function, k::Int64)
    if Bool(rand(Bernoulli(sticks(d, k))))
        return k
    else
        return pickStick(d, sticks, k+1)
    end
end

function makeSticks(d::DiscreteRandomProbabilityMeasure)
    function sticks(d::DiscreteRandomProbabilityMeasure, index::Int64)
        if length(d.sticks) <= index
            push!(d.sticks, sampleStick(d))
        end
        return d.sticks[index]
    end
    return pickStick(d::DiscreteRandomProbabilityMeasure, sticks, 1)
end

function sampleStick(d::PoissonKingmanMeasure)
    J = sampleWeight(d)
    V = J / d.T_surplus
    d.T_surplus = d.T_surplus - J
    return V
end

# NOTE: Explicit recursion can be slower, especially when variance is big
# function Distributions.rand(d::DiscreteRandomProbabilityMeasure)
#     index = makeSticks(d)
#     if !haskey(d.atoms, index)
#         d.atoms[index] = rand(d.base)
#     end
#     # println(d.atoms[index])
#     return d.atoms[index]
# end

# NOTE: Use metaprograming to generate this function ?
# function Distributions.rand(d::DistributionOnDistributions)
#     field_names = fieldnames(d)
#     parameters = Array(Float64,0)
#     for i in 1:length(field_names)-1
#         push!(parameters, getfield(d, field_names[i]))
#     end
#     DiscreteRandomProbabilityMeasureName = eval(Symbol(string(split(string(typeof(d)), "Turing.")[2],"sample")))
#     return DiscreteRandomProbabilityMeasureName(parameters..., d.base)
# end

function sampleWeight(d::NormalizedRandomMeasure)
    return sampleStick(d) * d.T_surplus
end

function Distributions.rand(d::NormalizedRandomMeasure)
    u = rand()
    thresh = 1 - d.T_surplus
    if u < thresh
        index = 1; c = d.lengths[1]
        while c < u
          c += d.lengths[index += 1]
        end
        return d.atoms[index]
    else
        J = sampleWeight(d)
        push!(d.lengths, J)
        d.T_surplus = d.T_surplus - J
        atom = rand(d.base)
        push!(d.atoms, atom)
        return atom
    end
end

function Distributions.rand(d::PoissonKingmanMeasure)
    u = rand()
    thresh = 1 - d.T_surplus/d.T
    if u < thresh
        index = 1; c = d.lengths[1]
        while c < u
            c += d.lengths[index += 1]
        end
        return d.atoms[index]
    else
        J = sampleWeight(d)
        push!(d.lengths, J/d.T)
        d.T_surplus = d.T_surplus - J
        atom = rand(d.base)
        push!(d.atoms, atom)
        return atom
    end
end