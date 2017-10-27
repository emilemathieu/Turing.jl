abstract DistributionOnDistributions      <: Distribution
abstract DiscreteRandomProbabilityMeasure <: DiscreteMultivariateDistribution
abstract NormalizedRandomMeasure          <: DiscreteRandomProbabilityMeasure
abstract PoissonKingmanMeasure            <: DiscreteRandomProbabilityMeasure
abstract NormalizedRandomMeasureRec       <: DiscreteRandomProbabilityMeasure
abstract PoissonKingmanMeasureRec         <: DiscreteRandomProbabilityMeasure

### DiscreteRandomProbabilityMeasure

@inline vectorize(d::DiscreteRandomProbabilityMeasure, r::Float64) = Vector{Real}([r])
@inline vectorize(d::DiscreteRandomProbabilityMeasure, r::Array{Float64,1}) = Vector{Real}(r)

Distributions.logpdf{T<:Real}(d::DiscreteRandomProbabilityMeasure, x::T) = zero(x)

function pickStick(d::DiscreteRandomProbabilityMeasure, sticks::Function, k::Int64)
    j = k
    while true
        if Bool(rand(Bernoulli(sticks(d, j))))
            return j
        else
            j += 1
        end
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

function sampleStick(d::Union{PoissonKingmanMeasure,PoissonKingmanMeasureRec})
    J = sampleWeight(d)
    V = J / d.T_surplus
    d.T_surplus = d.T_surplus - J
    return V
end

# NOTE: Explicit recursion can be slower, especially when variance is big
# function Distributions.rand(d::DiscreteRandomProbabilityMeasure)
function Distributions.rand(d::Union{NormalizedRandomMeasureRec,PoissonKingmanMeasureRec})
    index = makeSticks(d)
    if !haskey(d.atoms, index)
        d.atoms[index] = rand(d.base)
    end
    return d.atoms[index]
end

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
        index = wsample(d.weights)
        return d.atoms[index]
    else
        J = sampleWeight(d)
        push!(d.weights, J)
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
        index = wsample(d.weights)
        return d.atoms[index]
    else
        J = sampleWeight(d)
        push!(d.weights, J/d.T)
        d.T_surplus = d.T_surplus - J
        atom = rand(d.base)
        push!(d.atoms, atom)
        return atom
    end
end
