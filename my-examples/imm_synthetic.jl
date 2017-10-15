using Turing
using Base.Test
include("plot.jl")
include("ess.jl")

# model to generate synthetic data; treats y as unobserved variables

@model infiniteMixturesim(N) = begin
  s ~ Gamma(10.0, 10.0)
  alpha = 2.0
  P = rand(DP(alpha, Normal(0, sqrt(10.0))))

  x = tzeros(N); y = tzeros(N)
  for i in 1:N
    x[i] ~ P
    y[i] ~ Normal(x[i], 1.0/sqrt(s))
  end
end

T = 10
varInfo_sim = infiniteMixturesim(T)()
x_truth = zeros(T)
data = zeros(T)
offset = 1 # /!\ number of random variables befre x[1]; here only s
s_truth = varInfo_sim[varInfo_sim.vns[1]]
for i in 1:T
  x_truth[i] = varInfo_sim[varInfo_sim.vns[(i-1)*2+1+offset]]
  data[i] = varInfo_sim[varInfo_sim.vns[(i-1)*2+2+offset]]
end

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
mixtureComponents = computeMixtureComponents(length(data), infiniteMixtur(data2), sampler, :x)

@test mean(mixtureComponents[2,:] .== mixtureComponents[4,:]) >= 0.6
@test mean(mixtureComponents[1,:] .== mixtureComponents[5,:]) >= 0.6
@test mean(mixtureComponents[13,:] .== mixtureComponents[20,:]) <= 0.2
