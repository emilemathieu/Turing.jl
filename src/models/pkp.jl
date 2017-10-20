type mLogBetaPK   <:  DistributionOnDistributions
    alpha         ::  Float64
    beta          ::  Float64
    base          ::  Array{Distribution}
    mLogBetaPK(alpha::Float64, beta::Float64, base) = begin
        new(alpha, beta, isa(base, Array) ? base : [base])
    end
end

function Distributions.rand(d::mLogBetaPK)
    return mLogBetaPKsample(d.alpha, d.beta, d.base)
end

type mLogBetaPKsample <: PoissonKingmanMeasure
    alpha         ::  Float64
    beta          ::  Float64
    base          ::  Array{Distribution}
    atoms         ::  Vector{Any}
    lengths       ::  Vector{Float64}
    T             ::  Float64
    T_surplus     ::  Float64
    mLogBetaPKsample(alpha::Float64, beta::Float64, base) = begin
        T = rand(Beta(alpha, beta))
        new(alpha, beta, base, [], Array(Float64,0), T, T)
    end
end

function rejectionSampler(f, g, M)
    u = rand(Uniform(0, 1))
    y = rand(g)
    iter = 1
    while(u >= f(y) / (M * pdf(g, y)))
        u = rand(Uniform(0, 1))
        y = rand(g)
        iter += 1
    end
    return y
end

function logBetaPKDentity(b, t, s)
    return (1 - exp(-b*s)) / (1 - exp(-s)) * (1 - exp(s-t))^(b-1) / (1 - exp(-t))^(b-1)# / t
end
function makeLogBetaPKDentity(b, t)
    return (s) -> logBetaPKDentity(b, t, s)
end
proposal = (v) -> Uniform(0, v)

function sampleWeight(d::mLogBetaPKsample)
    return rejectionSampler(makeLogBetaPKDentity(d.beta, d.T_surplus), proposal(d.T_surplus), d.beta*d.T_surplus)
end
