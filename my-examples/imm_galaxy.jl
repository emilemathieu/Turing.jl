using Turing
using Turing:resampleSystematic
using Plots
include("ess.jl")

# Galaxy dataset
data = [9172.0;9350.0;9483.0;9558.0;9775.0;10227.0;10406.0;16084.0;16170.0;18419.0;18552.0;18600.0;18927.0;19052.0;19070.0;19330.0;19343.0;19349.0;19440.0;19473.0;19529.0;19541.0;19547.0;19663.0;19846.0;19856.0;19863.0;19914.0;19918.0;19973.0;19989.0;20166.0;20175.0;20179.0;20196.0;20215.0;20221.0;20415.0;20629.0;20795.0;20821.0;20846.0;20875.0;20986.0;21137.0;21492.0;21701.0;21814.0;21921.0;21960.0;22185.0;22209.0;22242.0;22249.0;22314.0;22374.0;22495.0;22746.0;22747.0;22888.0;22914.0;23206.0;23241.0;23263.0;23484.0;23538.0;23542.0;23666.0;23706.0;23711.0;24129.0;24285.0;24289.0;24366.0;24717.0;24990.0;25633.0;26960.0;26995.0;32065.0;32789.0;34279.0]
data /= 1e4
mu_0 = mean(data); sigma_0 = 1/sqrt(0.635); sigma_1 = sigma_0/15
# data -= mean(data) # mu_0 = 0.0; sigma_1 = .1; sigma_0 = 4*sigma_1;

function Base.convert(::Type{Float64}, x::Array{Float64,1})
  if length(x) > 1 error("FAIL: Cannot `convert` an object of type Array{Float64,1} to an object of type Float64") end
  return x[1]
end

@model infiniteMixture(y, RPM) = begin
  N = length(y)
  P = rand(RPM)
  # P = rand(DP(2.0, Normal(mu_0, sigma_0)))
  # P = rand(PYP(0.5, 1.0, Normal(mu_0, sigma_0)))
  # P = rand(NGGP(0.5, .01, Normal(mu_0, sigma_0)))

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

function sweeps_predictive_density(xaxis, SMC_sweeps, RPM)
    yaxis = zeros(length(xaxis))
    for i in 1:SMC_sweeps
        println("SMC sweep: ", i)
        permutation = randperm(length(data))
        results = sample(infiniteMixture(data[permutation], RPM), sampler)
        yaxis += compute_predictive_density(results, xaxis)
    end
    yaxis /= SMC_sweeps
    return yaxis
end

xaxis = linspace(0.8*minimum(data),1.2*maximum(data),200)
SMC_sweeps = 5
yaxisPYP = sweeps_predictive_density(xaxis, SMC_sweeps, PYP(0.5, 1.0, Normal(mu_0, sigma_0)))
yaxisNIGP = sweeps_predictive_density(xaxis, SMC_sweeps, NGGP(0.5, .01, Normal(mu_0, sigma_0)))

using Plots
font = Plots.font("Helvetica", 16)
pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font, size=(750, 375))
# pyplot()


# yaxis = compute_predictive_density(results, xaxis)
plot(xaxis, yaxisPYP/5, linewidth=2, label="Pitman-Yor(0.5, 1)", xticks=[], yticks=[])
plot!(xaxis, yaxisNIGP/5, linewidth=2, color=:green, label="NIGP(0.01)", xticks=[], yticks=[])
bar!(hist(data, 100), alpha=.5, linewidth=0, color=:brown, label="Data", xticks=[], yticks=[])
ylabel!("Probability density", fontsize=16)
savefig("predictive-density.pdf")
