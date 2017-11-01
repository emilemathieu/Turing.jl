using Turing
using Turing:resampleSystematic
include("plot.jl")
include("ess.jl")
using PlotlyJS

# Galaxy dataset
data = [9172.0;9350.0;9483.0;9558.0;9775.0;10227.0;10406.0;16084.0;16170.0;18419.0;18552.0;18600.0;18927.0;19052.0;19070.0;19330.0;19343.0;19349.0;19440.0;19473.0;19529.0;19541.0;19547.0;19663.0;19846.0;19856.0;19863.0;19914.0;19918.0;19973.0;19989.0;20166.0;20175.0;20179.0;20196.0;20215.0;20221.0;20415.0;20629.0;20795.0;20821.0;20846.0;20875.0;20986.0;21137.0;21492.0;21701.0;21814.0;21921.0;21960.0;22185.0;22209.0;22242.0;22249.0;22314.0;22374.0;22495.0;22746.0;22747.0;22888.0;22914.0;23206.0;23241.0;23263.0;23484.0;23538.0;23542.0;23666.0;23706.0;23711.0;24129.0;24285.0;24289.0;24366.0;24717.0;24990.0;25633.0;26960.0;26995.0;32065.0;32789.0;34279.0]
data /= 1e4
mu_0 = mean(data); sigma_0 = 1/sqrt(0.635); sigma_1 = sigma_0/10
# data -= mean(data) # mu_0 = 0.0; sigma_1 = .1; sigma_0 = 4*sigma_1;

function Base.convert(::Type{Float64}, x::Array{Float64,1})
  if length(x) > 1 error("FAIL: Cannot `convert` an object of type Array{Float64,1} to an object of type Float64") end
  return x[1]
end

@model infiniteMixture(y) = begin
  N = length(y)
  # P = rand(DP(10.0, Normal(mu_0, sigma_0)))
  P = rand(NGGP(0.5, 1.0, Normal(mu_0, sigma_0)))

  x = tzeros(N)
  @inbounds for i in 1:N
    P = deepcopy(P)
    x[i] ~ P
    y[i] ~ Normal(x[i], sigma_1)
  end
  weights = zeros(Float64, N); atoms = zeros(Float64, N);
  weights[1:length(P.weights)] = P.weights
  atoms[1:length(P.atoms)] = P.atoms
  w ~ Dirac(weights)
  a ~ Dirac(atoms)
end

sampler = SMC(1000, resampleSystematic, .5, false, Set(), 0)
# sampler = CSMC(100, 20)
# permutation = randperm(length(data))
# results = sample(infiniteMixture(data[permutation]), sampler)
# nb_clusters = [length(unique(xt)) for xt in results[:x]]

function compute_predictive_density(results, xaxis)
    a = results[:a]; w = results[:w]
    particles_w = [sample.weight for sample in results.value2]
    M = length(a)
    y = zeros(M, length(xaxis))
    @inbounds for i in 1:length(xaxis)
      println(i)
      xi = xaxis[i]
      @inbounds for j in 1:M
          K = find(a[j] .== 0.0)[1] - 1
          xj = a[j][1:K]
          pj = w[j][1:K]
          @inbounds for k in 1:K
            y[j,i] += pj[k]*pdf(Normal(xj[k], sigma_1), xi)
          end
      end
    end
    yaxis = sum(y.*particles_w,1)[1,:]*30
    # yaxis = mean(y, 1)[1,:]*30
    return yaxis
end

xaxis = linspace(0.7*minimum(data),1.0*maximum(data),80)

SMC_sweeps = 5
yaxis = zeros(length(xaxis))
for i in 1:SMC_sweeps
    println("SMC sweep: ", i)
    permutation = randperm(length(data))
    results = sample(infiniteMixture(data[permutation]), sampler)
    yaxis += compute_predictive_density(results, xaxis)
end
yaxis /= SMC_sweeps

# yaxis = compute_predictive_density(results, xaxis)
trace1 = scatter(;x=xaxis, y=yaxis, mode="lines+markers")
trace2 = histogram(x=data, opacity=0.75, name="")
p = plot([trace1, trace2])





function linescatter(y0,y1,x0=1:length(y0),x1=1:length(y1),yname="y",xname="x", name1="", name2="")
    trace1 = scatter(;x=x0, y=y0, mode="lines+markers",name=name1)
    trace2 = scatter(;x=x1, y=y1, mode="lines+markers",name=name2)
    layout = Layout(plot_bgcolor="white",paper_bgcolor="white",yaxis=attr(title=yname,showgrid=false,showline=false),xaxis=attr(title=xname,showgrid=false))
    plot([trace1, trace2], layout)
end

function plot_2_components(results, index1, index2)
    mixtureComponentsRes = results[:x]
    T = 82
    mixtureComponents = zeros(T, length(mixtureComponentsRes))
    for j in 1:size(mixtureComponents,2)
        mixtureComponents[:,j] = mixtureComponentsRes[j]
    end
    linescatter(mixtureComponents[index1,:],mixtureComponents[index2,:],1:length(mixtureComponentsRes),1:length(mixtureComponentsRes),"X(t)","x", "X(t=1)","X(t=82)")
end
# plot_2_components(results, 1, 82)

function compute_coclustering(results)
    T = 82
    coclustering = zeros(T, T)
    particles_w = [sample.weight for sample in results.value2]
    @inbounds for k in 1:length(results[:x])
      if particles_w[k] > 1e-3
          println(k)
          sample = results[:x][k]
          @inbounds for i in 1:T
            @inbounds for j in 1:T
              coclustering[i, j] += particles_w[k]*(sample[i] == sample[j])
            end
          end
      end
    end
    return coclustering
end
# coclustering = compute_coclustering(results)
# plot(heatmap(z=coclustering))
