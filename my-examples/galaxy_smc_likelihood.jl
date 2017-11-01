using Turing

# Galaxy dataset
data = [9172.0;9350.0;9483.0;9558.0;9775.0;10227.0;10406.0;16084.0;16170.0;18419.0;18552.0;18600.0;18927.0;19052.0;19070.0;19330.0;19343.0;19349.0;19440.0;19473.0;19529.0;19541.0;19547.0;19663.0;19846.0;19856.0;19863.0;19914.0;19918.0;19973.0;19989.0;20166.0;20175.0;20179.0;20196.0;20215.0;20221.0;20415.0;20629.0;20795.0;20821.0;20846.0;20875.0;20986.0;21137.0;21492.0;21701.0;21814.0;21921.0;21960.0;22185.0;22209.0;22242.0;22249.0;22314.0;22374.0;22495.0;22746.0;22747.0;22888.0;22914.0;23206.0;23241.0;23263.0;23484.0;23538.0;23542.0;23666.0;23706.0;23711.0;24129.0;24285.0;24289.0;24366.0;24717.0;24990.0;25633.0;26960.0;26995.0;32065.0;32789.0;34279.0]
data /= 1e4
data -= mean(data)

mu_0 = 0.0; sigma_1 = 10; sigma_0 = 4*sigma_1;

# unexplained, but potentially important code
@inline realpart(r::Real)             = r
@inline realpart(d::ForwardDiff.Dual) = d.value

# model
@model PYPGalaxy(y, alpha, theta, recursive) = begin
  N = length(y)
  P = rand(PYP(alpha,
               theta,
               Normal(mu_0, sigma_0),
               recursive))

  x = zeros(N)
  for i in 1:N
    P_copy = deepcopy(P)
    x[i] ~ P_copy
    y[i] ~ Normal(x[i], sigma_1)
  end
end

# hyperparameters
alpha = 0.1
theta = 0.25

# sampling using SMC
w = zeros(100, 2)
t = zeros(100, 2)
for i=1:100
  println("Iteration ", i)
  for (j, recursive) = enumerate([true, false])
    tic()
    sampler = SMC(1000)
    results = sample(PYPGalaxy(data, alpha, theta, recursive), sampler)
    w[i,j] = log(results.weight)
    t[i,j] = toc()
  end
end

println(mean(w, 1))
println(std(w, 1))
println(mean(t, 1))
println(std(t, 1))
