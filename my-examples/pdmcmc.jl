using Turing
using Gallium

@model NLSSM1(y) = begin
  N = length(y)
  sigma2_v = sigma2_w = 10
  # sigma2_v= 10; sigma2_w = 1

  x = Vector{Any}(N)
  x[1] ~ Normal(0, sqrt(5))
  y[1] ~ Normal(x[1]^2/20, sqrt(sigma2_w))
  for i in 2:N
    x[i] ~ Normal(x[i-1]/2 + 25*x[i-1]/(1+x[i-1]^2) + 8*cos(1.2*i), sqrt(sigma2_v))
    y[i] ~ Normal(x[i]^2/20, sqrt(sigma2_w))
  end
end

sampler = PDMCMC(1, 1.0)

# results = sample(NLSSM1([1.0,2.2,1.5]), sampler)
@step sample(NLSSM1([1.0,2.2,1.5]), sampler)
