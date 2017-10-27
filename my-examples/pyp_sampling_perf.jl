using Turing
# include("plot.jl")

# Compare performance between (stochastic & unbounded) recursive style and
# Sise-biased sampling style
# /!\ DP2 and mLogBetaPK2 do not exist anymore; Go to src/models/rpm.jl
# comment Distributions.rand(d::NormalizedRandomMeasure) & Distributions.rand(d::PoissonKingmanMeasure)
# and uncomment Distributions.rand(d::DiscreteRandomProbabilityMeasure)

alphas = [0.1 0.5 0.75]
thetas = [1.0 10.0]
samples = [1 10 100 1000 10000]
recursives = [true false]

timings = []
for recursive = recursives
  for alpha = alphas
    for theta = thetas
      for sample_size = samples
        sps = zeros(sample_size)
        P = rand(PYP(alpha, theta, Normal(0, 1), recursive))
        for i=1:sample_size
          sps[i] = rand(P)
        end
        append!(timings, [(recursive, alpha, theta, sample_size, length(P.atoms))])
      end
    end
  end
end



println(timings)
