using Turing
include("plot.jl")
include("ess.jl")
# using Base.Test
# srand(125)

x_true = [-5.82973,-0.00683593,-1.03445,-1.03445,-5.82973,-1.03445,-5.82973,-0.00683593,-0.00683593,-5.82973,-5.82973,0.251331,-0.00683593,-5.82973,-1.03445,-5.82973,-0.00683593,-5.82973,-0.00683593,-2.96891]
s_true = 5.04821
data = [-5.181,-0.093952,-0.839613,-1.47678,-5.28014,-1.66853,-6.43058,-0.08569,-0.257869,-5.79966,-5.7677,0.487894,-0.119956,-6.21431,-0.604658,-5.98347,-0.379566,-5.37807,-0.480076,-3.60479]
# linescatter(x_true, data)
# data = zeros(length(x_true))
# for i in 1:length(x_true)
#     data[i] = rand(Normal(x_true[i], 1.0/sqrt(s_true)))
# end

@model infiniteMixture(y) = begin
  N = length(y)
  s ~ Gamma(10.0, 10.0)
  alpha = 2.0
  P = rand(DP(alpha, Normal(0, sqrt(10.0))))

  x = tzeros(N)
  for i in 1:N
    x[i] ~ P
    y[i] ~ Normal(x[i], 1.0/sqrt(s))
  end
end

# N_samples = 100
# N_particles = 10

# sampler = Gibbs(N_samples, CSMC(N_particles, 1, :x), HMC(1, 0.1, 3, :s))
# sampler = Gibbs(N_samples, CSMC(N_particles, 1, :x), HMC(1, 0.1, 3, :s))
sampler = IPMCMC(100, 40, 10, 5)
results = sample(infiniteMixture(data), sampler)

M = 1
mixtureComponentsESS = zeros(M, length(data))
for l = 1:M
    println(l)
    shuffled_data = data[randperm(length(data))]
    results = sample(infiniteMixture(shuffled_data), sampler)
    # results = sample(infiniteMixture(data), sampler)
    mixtureComponentsRes = results[:x]
    mixtureComponents = zeros(length(data), length(mixtureComponentsRes))

    for j in 1:size(mixtureComponents,2)
        mixtureComponents[:,j] = mixtureComponentsRes[j]
    end
    for i in 1:size(mixtureComponents,1)
        mixtureComponentsESS[l,i] = ess_factor(mixtureComponents[i,:])
        if isnan(mixtureComponentsESS[l,i])
            mixtureComponentsESS[l,i] = 0.0
        end
    end
end
mixtureComponentsESSquartiles = zeros(3, length(data))
for i in 1:length(data)
    mixtureComponentsESSquartiles[:, i] = quantile(mixtureComponentsESS[:,i], [.25, .5, .75])
end

# linescatter(mean(mixtureComponentsESS, 1)[1,:])
ESSPlotVariance(mixtureComponentsESSquartiles)


















# mixtureComponentsRes = results[:x]
# mixtureComponents = zeros(length(data), length(mixtureComponentsRes))
# for j in 1:size(mixtureComponents,2)
#     mixtureComponents[:,j] = mixtureComponentsRes[j]
# end

# @test mean(mixtureComponents[2,:] .== mixtureComponents[4,:]) >= 0.6
# @test mean(mixtureComponents[1,:] .== mixtureComponents[5,:]) >= 0.6
# @test mean(mixtureComponents[13,:] .== mixtureComponents[20,:]) <= 0.2
