using Turing
include("plot.jl")

# Compare performance between (stochastic & unbounded) recursive style and
# Sise-biased sampling style
# /!\ DP2 and mLogBetaPK2 do not exist anymore; Go to src/models/rpm.jl
# comment Distributions.rand(d::NormalizedRandomMeasure) & Distributions.rand(d::PoissonKingmanMeasure)
# and uncomment Distributions.rand(d::DiscreteRandomProbabilityMeasure)

m = 0.0
s = 1.0
a = 10.0
b = 2.0
N = 1000
K = 10000

function sample_RPM(alpha, beta, K, N)
  for k in 1:K
    samples = zeros(N)
    P = rand(DP(alpha, Normal(m, 1.0/sqrt(s))))
    # P = rand(mLogBetaPK(alpha, beta, Normal(m, 1.0/sqrt(s))))
    for i in 1:N
      samples[i] = rand(P)
    end
  end
end

function sample_RPM2(alpha, beta, K, N)
  for k in 1:K
    samples2 = zeros(N)
    P = rand(DP2(alpha, Normal(m, 1.0/sqrt(s))))
    # P = rand(mLogBetaPK2(alpha, beta, Normal(m, 1.0/sqrt(s))))
    for i in 1:N
      samples2[i] = rand(P)
    end
  end
end

# Trigger compilation
sample_RPM(a, b, K, N)
sample_RPM2(a, b, K, N)

as = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
time = zeros(length(as))
time2 = zeros(length(as))
for j in 1:length(as)
  println(as[j])
  a = as[j]
  time[j] = @elapsed sample_RPM(a, b, K, N)
  time2[j] = @elapsed sample_RPM2(a, b, K, N)
end
linescatter(time,time2,as,as,"time (s)","alpha","stick-breaking","size-biased")
