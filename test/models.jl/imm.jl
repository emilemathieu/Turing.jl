using Turing
using Base.Test
srand(125)

data = [-3.61133,0.985144,-0.27336,0.985144,-3.61133,-0.283858,-0.27336,0.985144,1.91394,-3.61133,1.91394,-3.61133,4.58216,-0.27336,-3.61133,1.91394,-0.283858,0.985144,-0.27336,-4.2654]

@model infiniteMixture(y) = begin
  N = length(y)
  s ~ Gamma(1.0, 10.0)
  alpha = 2.0
  P = rand(DP(alpha, Normal(0, sqrt(30/s))))

  x = zeros(N)
  for i in 1:N
    x[i] ~ P
    y[i] ~ Normal(x[i], 1.0/sqrt(s))
  end
end

N_samples = 200
N_particles = 30

sampler = Gibbs(N_samples, CSMC(N_particles, 1, :x), HMC(1, 0.1, 3, :s))
results = sample(infiniteMixture(data), sampler)

mixtureComponentsRes = results[:x]
mixtureComponents = zeros(length(data), length(mixtureComponentsRes))
for j in 1:size(mixtureComponents,2)
    mixtureComponents[:,j] = mixtureComponentsRes[j]
end

@test mean(mixtureComponents[2,:] .== mixtureComponents[4,:]) >= 0.6
@test mean(mixtureComponents[1,:] .== mixtureComponents[5,:]) >= 0.6
@test mean(mixtureComponents[13,:] .== mixtureComponents[20,:]) <= 0.2
