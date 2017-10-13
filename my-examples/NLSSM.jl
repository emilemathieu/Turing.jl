using Turing
include("plot.jl")
include("ess.jl")

###################### WITHOUT global parameters #########################

## Similator
@model NLSSM1sim(N) = begin
  a = b = 0.01

  sigma2_v = sigma2_w = 10
  # sigma2_v= 10; sigma2_w = 1

  x = tzeros(N)
  y = tzeros(N)
  x[1] ~ Normal(0, sqrt(5))
  y[1] ~ Normal(x[1]^2/20, sqrt(sigma2_w))
  for i in 2:N
    x[i] ~ Normal(x[i-1]/2 + 25*x[i-1]/(1+x[i-1]^2) + 8*cos(1.2*i), sqrt(sigma2_v))
    y[i] ~ Normal(x[i]^2/20, sqrt(sigma2_w))
  end
end

T = 10
varInfo_sim1 = NLSSM1sim(T)()
x_truth1 = zeros(T)
data1 = zeros(T)
for i in 1:T
  x_truth1[i] = varInfo_sim1[varInfo_sim1.vns[(i-1)*2+1]]
  data1[i] = varInfo_sim1[varInfo_sim1.vns[(i-1)*2+2]]
end

## Model
@model NLSSM1(y) = begin
  N = length(y)
  sigma2_v = sigma2_w = 10
  # sigma2_v= 10; sigma2_w = 1

  x = tzeros(N)
  x[1] ~ Normal(0, sqrt(5))
  y[1] ~ Normal(x[1]^2/20, sqrt(sigma2_w))
  for i in 2:N
    x[i] ~ Normal(x[i-1]/2 + 25*x[i-1]/(1+x[i-1]^2) + 8*cos(1.2*i), sqrt(sigma2_v))
    y[i] ~ Normal(x[i]^2/20, sqrt(sigma2_w))
  end
end

N_samples = 50
N_particles = 50
pimh = PMMH(N_samples, SMC(N_particles, :x))
# results1_pimh = sample(NLSSM1(data1), pimh)

######################### WITH global parameters ############################

## Simulator
@model NLSSM2sim(N) = begin
  a = b = 0.01

  sigma2_v = 10; sigma2_w = 1

  x = tzeros(N)
  y = tzeros(N)
  x[1] ~ Normal(0, sqrt(5))
  y[1] ~ Normal(x[1]^2/20, sqrt(sigma2_w))
  for i in 2:N
    x[i] ~ Normal(x[i-1]/2 + 25*x[i-1]/(1+x[i-1]^2) + 8*cos(1.2*i), sqrt(sigma2_v))
    y[i] ~ Normal(x[i]^2/20, sqrt(sigma2_w))
  end
end

T = 10
varInfo_sim2 = NLSSM2sim(T)()
x_truth2 = zeros(T)
data2 = zeros(T)
for i in 1:T
  x_truth2[i] = varInfo_sim2[varInfo_sim2.vns[(i-1)*2+1]]
  data2[i] = varInfo_sim2[varInfo_sim2.vns[(i-1)*2+2]]
end

## Model
@model NLSSM2(y) = begin
  N = length(y)
  a = b = 0.01

  sigma2_v ~ InverseGamma(a, b)
  sigma2_w ~ InverseGamma(a, b)

  x = tzeros(N)
  x[1] ~ Normal(0, sqrt(5))
  y[1] ~ Normal(x[1]^2/20, sqrt(sigma2_w))
  for i in 2:N
    x[i] ~ Normal(x[i-1]/2 + 25*x[i-1]/(1+x[i-1]^2) + 8*cos(1.2*i), sqrt(sigma2_v))
    y[i] ~ Normal(x[i]^2/20, sqrt(sigma2_w))
  end
end

q_sigma2_v = (s) -> Normal(s, sqrt(0.15))
q_sigma2_w = (s) -> Normal(s, sqrt(0.08))

N_samples = 100
N_particles = 100
pmmh = PMMH(N_samples, SMC(N_particles, :x), (:sigma2_v, q_sigma2_v), (:sigma2_w, q_sigma2_w))
pg = PG(N_particles, N_samples)
# results = sample(NLSSM2(data2), pg)


# sampler = Gibbs(N_samples, CSMC(50, 1, :x), HMC(1, 0.2, 3, :s))
# sampler = IPMCMC(100, 10, 8, 4)
ipmcmc = IPMCMC(N_particles, Int(N_samples/2), 4, 2)
# impmcmc = IPMCMC(N_particles, Int(N_samples/2), 4, 2, HMC(1, 0.2, 3, :sigma2_v, :sigma2_w), :x)
# results = sample(NLSSM2(data2), impmcmc)


########### Analyzing

# mixtureComponentsESS = zeros(T)
# mixtureComponentsRes = results[:x]
# mixtureComponents = zeros(T, length(mixtureComponentsRes))
#
# for j in 1:size(mixtureComponents,2)
#     mixtureComponents[:,j] = mixtureComponentsRes[j]
# end
# for i in 1:size(mixtureComponents,1)
#     mixtureComponentsESS[i] = ess_factor(mixtureComponents[i,:])
#     if isnan(mixtureComponentsESS[i])
#         mixtureComponentsESS[i] = 0.0
#     end
# end

M = 10
mixtureComponentsESS = zeros(M, T)
for l = 1:M
    println(l)
    results = sample(NLSSM2(data2), ipmcmc)
    mixtureComponentsRes = results[:x]
    mixtureComponents = zeros(T, length(mixtureComponentsRes))

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
mixtureComponentsESSquartiles = zeros(3, T)
for i in 1:T
    mixtureComponentsESSquartiles[:, i] = quantile(mixtureComponentsESS[:,i], [.25, .5, .75])
end


############## Plots
# linescatter(mixtureComponentsESS)
ESSPlotVariance(mixtureComponentsESSquartiles, "PG")
