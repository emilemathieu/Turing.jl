using Turing
using Turing: VarInfo, Sampler, resampleSystematic, view, getidcs, is_inside, realpart, invlink, randuni
using Gallium

immutable UrnDP <: ContinuousUnivariateDistribution
    alpha::Float64
    T_surplus::Float64
end
Distributions.rand(d::UrnDP) = d.T_surplus*rand(Beta(1, d.alpha))
Distributions.logpdf{T<:Real}(d::UrnDP, x::T) = logpdf(Beta(1, d.alpha),x/d.T_surplus)
Distributions.minimum(d::UrnDP) = 0.0
Distributions.maximum(d::UrnDP) = d.T_surplus
# init(dist::UrnDP) = rand(dist)

srand(100)

data = [9172.0;9350.0;9483.0;9558.0;9775.0;10227.0;10406.0;16084.0;16170.0;18419.0;18552.0;18600.0;18927.0;19052.0;19070.0;19330.0;19343.0;19349.0;19440.0;19473.0;19529.0;19541.0;19547.0;19663.0;19846.0;19856.0;19863.0;19914.0;19918.0;19973.0;19989.0;20166.0;20175.0;20179.0;20196.0;20215.0;20221.0;20415.0;20629.0;20795.0;20821.0;20846.0;20875.0;20986.0;21137.0;21492.0;21701.0;21814.0;21921.0;21960.0;22185.0;22209.0;22242.0;22249.0;22314.0;22374.0;22495.0;22746.0;22747.0;22888.0;22914.0;23206.0;23241.0;23263.0;23484.0;23538.0;23542.0;23666.0;23706.0;23711.0;24129.0;24285.0;24289.0;24366.0;24717.0;24990.0;25633.0;26960.0;26995.0;32065.0;32789.0;34279.0]
data /= 1e4
mu_0 = mean(data); sigma_0 = 1/sqrt(0.635); sigma_1 = sigma_0/15

@model infiniteMixture(y) = begin
  N = length(y)
  H = Normal(mu_0, sigma_0)
  # alpha = 10

  a = tzeros(N); p = tzeros(N); z = tzeros(Int, N)
  # a = tzeros(ForwardDiff.Dual,N); p = tzeros(ForwardDiff.Dual,N); z = tzeros(Int, N)
  # a = Array{Dual}(N); p = Array{Dual}(N); z =  Array{Dual}(N)
  k = 0; T_surplus = 1.0
  for i in 1:N
    # println("k=",k)
    # println("kBIS=",sum(a .!= 0))
    # println("p: ", p)
    # println("p[1:k]: ", p[1:k])
    # println("T_surplus: ", T_surplus)
    # ps = k >= 1 ? vcat(p[1:k], T_surplus) : [Dual(T_surplus)]
    # println("T_surplusBIS=", 1 - sum(p[1:k]))
    ps = vcat(p[1:k], T_surplus)
    # println("ps: ", ps)
    z[i] ~ Categorical(ps)
    # println("z[i]=",z[i])
    if z[i] > k
      k = k + 1
      p[k] ~ UrnDP(2, T_surplus)
    #   println("p[",k,"]=", p[k])
      a[k] ~ H
      T_surplus -= p[k]
    end
    y[i] ~ Normal(a[realpart(z[i])], sigma_1)
  end
end
# sampler = SMC(2, resampleSystematic, .5, false, Set(), 0)
sampler = CSMC(10, 4, resampleSystematic, Set(), 0) # reproduce bug
# sampler = CSMC(15, 20, resampleSystematic, Set(), 0)
# sampler = Gibbs(3, CSMC(2, 1, :z),HMC(1,0.1,3,:p,:a))
# sampler = HMC(2,0.1,3)
permutation = randperm(length(data))
# @step sample(infiniteMixture(data[permutation]), sampler)
results = sample(infiniteMixture(data[permutation]), sampler)
# println([isnan(sum(sample[:z])) for sample in results])
# println([isnan(sum(sample[:a])) for sample in results])
# println([isnan(sum(sample[:p])) for sample in results])
println([(isnan(sum(sample[:a][sample[:z]])) || isnan(sum(sample[:p][sample[:z]]))) for sample in results])

function compute_predictive_density(results, xaxis)
    a = [sample[:a] for sample in results]
    w = [sample[:p] for sample in results]
    # particles_w = [sample.weight for sample in results]
    particles_w = ones(length(results))/length(results)
    M = length(a)
    y = zeros(M, length(xaxis))
    @inbounds for i in 1:length(xaxis)
      println(i)
      xi = xaxis[i]
      @inbounds for j in 1:M
          xj = a[j]
          pj = w[j]
          @inbounds for k in 1:length(xj)
            if !isnan(xj[k]) y[j,i] += pj[k]*pdf(Normal(xj[k], sigma_1), xi) end
          end
      end
    end
    yaxis = sum(y.*particles_w,1)[1,:]*15
    return yaxis
end

function sweeps_predictive_density(xaxis, SMC_sweeps)
    yaxis = zeros(length(xaxis))
    for i in 1:SMC_sweeps
        println("SMC sweep: ", i)
        permutation = randperm(length(data))
        results = sample(infiniteMixture(data[permutation]), sampler)
        yaxis += compute_predictive_density(results, xaxis)
    end
    yaxis /= SMC_sweeps
    return yaxis
end
#
# xaxis = linspace(0.7*minimum(data),1.15*maximum(data),80)
# SMC_sweeps = 3
# yaxis = sweeps_predictive_density(xaxis, SMC_sweeps)
# #
# using PlotlyJS
# trace1 = scatter(;x=xaxis, y=yaxis, mode="lines+markers")
# trace3 = histogram(x=data, opacity=0.75, name="")
# p = plot([trace1, trace3])

#if !isempty(vi) && ref_particle.vi.vns[end].sym == :a
#   println(map(e -> length(e)>0, [[length(particles[i].vi[vn])>0 for vn in particles[i].vi.vns if (vn.sym == :z && particles[i].vi[vn] == 9)] for i in 1:10]))
#   println([[!isnan(particles[i].vi[vn]) for vn in particles[i].vi.vns if (vn.sym == :a && vn.indexing == "[9]")][1] for i in 1:10])
#   println("ISSUE:",find((map(e -> length(e)>0, [[length(particles[i].vi[vn])>0 for vn in particles[i].vi.vns if (vn.sym == :z && particles[i].vi[vn] == 9)] for i in 1:10]) .== [[!isnan(particles[i].vi[vn]) for vn in particles[i].vi.vns if (vn.sym == :a && vn.indexing == "[9]")][1] for i in 1:10]) .== false))
#end
