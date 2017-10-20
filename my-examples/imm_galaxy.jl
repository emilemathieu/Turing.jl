using Turing
include("plot.jl")

# Galaxy dataset
data = [9172.0;9350.0;9483.0;9558.0;9775.0;10227.0;10406.0;16084.0;16170.0;18419.0;18552.0;18600.0;18927.0;19052.0;19070.0;19330.0;19343.0;19349.0;19440.0;19473.0;19529.0;19541.0;19547.0;19663.0;19846.0;19856.0;19863.0;19914.0;19918.0;19973.0;19989.0;20166.0;20175.0;20179.0;20196.0;20215.0;20221.0;20415.0;20629.0;20795.0;20821.0;20846.0;20875.0;20986.0;21137.0;21492.0;21701.0;21814.0;21921.0;21960.0;22185.0;22209.0;22242.0;22249.0;22314.0;22374.0;22495.0;22746.0;22747.0;22888.0;22914.0;23206.0;23241.0;23263.0;23484.0;23538.0;23542.0;23666.0;23706.0;23711.0;24129.0;24285.0;24289.0;24366.0;24717.0;24990.0;25633.0;26960.0;26995.0;32065.0;32789.0;34279.0]
data /= 1e4
meanMean = 2.173; meanPrecision = 0.635; precisionShape = 2.0;
precisionInvScaleAlpha = 0.4/2; precisionInvScaleBeta = 6.346
precisionInvScale = precisionInvScaleAlpha / precisionInvScaleBeta
# precisionInvScale ~ Beta(precisionInvScaleAlpha, precisionInvScaleBeta)

@inline realpart(r::Real)             = r
@inline realpart(d::ForwardDiff.Dual) = d.value

@model infiniteMixture(y) = begin
  N = length(y)
  s ~ Gamma(precisionShape, 1.0/precisionInvScale)
  # alpha ~ Gamma(1.0, 1.0)
  alpha = 1.5
  # P = rand(DP(alpha, Normal(meanMean, 1.0/sqrt(meanPrecision))))
  P = rand(mLogBetaPK(1.0, 2.0, Normal(meanMean, 1.0/sqrt(meanPrecision))))

  x = zeros(N)
  for i in 1:N
    x[i] ~ P
    y[i] ~ Normal(x[i], 1.0/sqrt(s))
  end
end

@model infiniteMixture2(y) = begin
  N = length(y)
  alpha = 1.5
  # P = rand(DP(alpha, [Normal(meanMean, 1.0/sqrt(meanPrecision)),Gamma(precisionShape, 1.0/precisionInvScale)]))
  P = rand(mLogBetaPK(1.0, 2.0, [Normal(meanMean, 1.0/sqrt(meanPrecision)),Gamma(precisionShape, 1.0/precisionInvScale)]))

  x = zeros(N,2)
  for i in 1:N
    x[i,1:2] ~ P
    y[i] ~ Normal(x[i,1], 1.0/sqrt(x[i,2]))
  end
end

# s_proposal = (s) -> Normal(s, 0.1*std(Gamma(precisionShape, 1.0/precisionInvScale)))

N_samples = 50
N_particles = 50

sampler = SMC(10)
# sampler = CSMC(20, 50)
# sampler = Gibbs(N_samples, CSMC(50, 1, :x), HMC(1, 0.2, 3, :s))
# sampler = PMMH(N_samples, SMC(40, :x), :s)
# sampler = PMMH(N_samples, SMC(N_particles, :x), (:s, s_proposal)) # 50 & 50
# sampler = Gibbs(5, IPMCMC(100, 1, 3, 1, :x), HMC(1, 0.2, 3, :s))
# sampler = IPMCMC(100, 10, 8, 4)
# sampler = IPMCMC(15, 25, 4, 2)
# sampler = IPMCMC(15, 25, 4, 2, HMC(1, 0.2, 3, :s), :x)
results = sample(infiniteMixture2(data), sampler)

# M = 20
# mixtureComponentsESSquartiles = computeMixtureComponentsESSquartiles(M, T, (infiniteMixture(data), sampler, :x)
# ESSPlotVariance(mixtureComponentsESSquartiles, "PG")

# linescatter(mixtureComponents[1,:],mixtureComponents[end,:],1:400,1:400,"X(t)","x", "X(t=1)","X(t=82)")
# linescatter(results[:s])
# plot_histogram(mixtureComponents[end,:])
# plot_histograms(mixtureComponents[1,:],mixtureComponents[end,:],"X(t=1)","X(t=82)")
# plot_histogram(results[:s])
