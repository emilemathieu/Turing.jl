using Turing
using Plots
using Gallium

@model caronFoxGGP(Z, D_alpha_star) = begin

  alpha ~ Exponential(1)
  W_alpha_star ~ T_NIGP(alpha)
  D_alpha_star ~ Poisson(W_alpha_star^2)
  
  weights = Dict()
  ncrm = NIGP(tau=57, W_alpha_star)
  Z_sample = matrix()
  for i=1:D_alpha_star
    outcome, stick_number, weight ~ ncrm
    weights[stick_number] = weight
    outcome2, stick_number2, weight2 ~ ncrm
    weights[stick_number2] = weight2
    Z_sample[stick_number, stick_number2] += 1
  end

  observe(Z = Z_sample)

  return weights

end

graph = [[2 1 1] [1 0 0] [0 0 0]]

sampler = SMC(100)
mdl = caronFoxGGP(graph)
samples = sample(mdl, sampler)
println(samples)
