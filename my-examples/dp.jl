using Turing
include("plot.jl")

data = [9172.0;9350.0;9483.0;9558.0;9775.0;10227.0;10406.0;16084.0;16170.0;18419.0;18552.0;18600.0;18927.0;19052.0;19070.0;19330.0;19343.0;19349.0;19440.0;19473.0;19529.0;19541.0;19547.0;19663.0;19846.0;19856.0;19863.0;19914.0;19918.0;19973.0;19989.0;20166.0;20175.0;20179.0;20196.0;20215.0;20221.0;20415.0;20629.0;20795.0;20821.0;20846.0;20875.0;20986.0;21137.0;21492.0;21701.0;21814.0;21921.0;21960.0;22185.0;22209.0;22242.0;22249.0;22314.0;22374.0;22495.0;22746.0;22747.0;22888.0;22914.0;23206.0;23241.0;23263.0;23484.0;23538.0;23542.0;23666.0;23706.0;23711.0;24129.0;24285.0;24289.0;24366.0;24717.0;24990.0;25633.0;26960.0;26995.0;32065.0;32789.0;34279.0]
data /= 1e4
meanMean = 2.17; meanPrecision = 0.63; precisionShape = 2.0; precisionInvScale = 0.2/6.34

# m = rand(Normal(meanMean, 1.0/sqrt(meanPrecision)))
# s = rand(Gamma(precisionShape, 1.0/precisionInvScale))
m = 0.0
s = 1.0
a = 10.0
b = 2.0
N = 1000
K = 10000

function sample_RPM(alpha, beta, K, N)
  for k in 1:K
    # println(k)
    samples = zeros(N)
    P = rand(DP(alpha, Normal(m, 1.0/sqrt(s))))
    # P = rand(mLogBetaPK(alpha, beta, Normal(m, 1.0/sqrt(s))))
    for i in 1:N
      samples[i] = rand(P)
    end
  end
end

function sample_RPM2(alpha, beta, K, N)
  for k in 1:K
    # println(k)
    samples2 = zeros(N)
    P = rand(DP2(alpha, Normal(m, 1.0/sqrt(s))))
    # P = rand(mLogBetaPK2(alpha, beta, Normal(m, 1.0/sqrt(s))))
    for i in 1:N
      samples2[i] = rand(P)
    end
  end
end

sample_RPM(a, b, K, N)
sample_RPM2(a, b, K, N)
# println(samples2)
# println(unique(samples2))
# sample_RPM(a, b, K, N);sample_RPM2(a, b, K, N);

# n = 1
# time = 0; time2 = 0
# for j in 1:n
#   time += @elapsed sample_RPM(a, b, K, N)
#   time2 += @elapsed sample_RPM2(a, b, K, N)
# end
# time /= n; time2 /= n
# println("time recursive: ", time)
# println("time non-recursive: ", time2)

as = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
time = zeros(length(as))
time2 = zeros(length(as))
for j in 1:length(as)
  println(as[j])
  a = as[j]
  time[j] = @elapsed sample_RPM(a, b, K, N)
  time2[j] = @elapsed sample_RPM2(a, b, K, N)
end
linescatter(time,time2,as,as,"time (s)","alpha","stick-breaking","size-biased")

# @profile sample_RPM2(a, b, K, N);Profile.print()
#
plot_histogram(samples)
plot_histogram(samples2)
