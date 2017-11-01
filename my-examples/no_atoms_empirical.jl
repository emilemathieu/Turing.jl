using Turing
# include("plot.jl")

# Compare performance between (stochastic & unbounded) recursive style and
# Sise-biased sampling style
# /!\ DP2 and mLogBetaPK2 do not exist anymore; Go to src/models/rpm.jl
# comment Distributions.rand(d::NormalizedRandomMeasure) & Distributions.rand(d::PoissonKingmanMeasure)
# and uncomment Distributions.rand(d::DiscreteRandomProbabilityMeasure)


## functions for calculating the probability distribution of the number of atoms generated
## by the recursive stick-breaking scheme for the Pitman-Yor process

using Memoize
@memoize function MaxAtomProb(n::Int,m::Int,alpha::AbstractFloat,theta::AbstractFloat)

  if(n==1)
    prob = 1 - exp(W_1m(m,alpha,theta))
  else
    prob = MaxAtomProb(n-1,m,alpha,theta) - MaxAtomProb(n-1,m,alpha,theta+1)*exp(W_1m(m,alpha,theta))
  end

  return prob

end

@memoize function W_1m(m::Int,alpha::AbstractFloat,theta::AbstractFloat)

  sum( log.(theta + (1:m).*alpha) - log.(theta + 1 + (0:(m-1)).*alpha) )

end

function factorialPower(x::AbstractFloat, n::Int, alpha::Real)
  s = 1
  for i=0:(n-1)
    s *= (x + i*alpha)
  end
  return s
end


function TheoryExpNoAtoms(N,M,alpha,theta, recursive)
  if recursive == false
    return [(factorialPower(theta + alpha, n, 1))/(alpha*factorialPower(theta+1, n-1, 1)) - (theta/alpha) for n=1:N]
  else
    P_nm = Array{Float64}(M,N)
    for m=1:M
      for n=1:N
        P_nm[m,n] = MaxAtomProb(n,m,alpha,theta)

      end
    end

    E_n = 1+sum(1 - P_nm, 1)

    return E_n
  end

end

function EmpiricalNoAtoms(N,n_samples,alpha,theta, recursive)
  s_nm = Array{Float64}(n_samples, N)
  for m=1:n_samples
    println("Iteration ", m)
    P = rand(PYP(alpha, theta, Normal(0, 1), recursive))
    #P = rand(DP(theta, Normal(0, 1), true))
    for n=1:N
      rand(P)
      if recursive
        s_nm[m,n] = length(P.sticks)
      else
        s_nm[m,n] = length(P.weights)
      end
    end
  end

  E_n = mean(s_nm, 1)
  s_n = 1.96*std(s_nm, 1)/sqrt(n_samples)

  return E_n, s_n

end


## tests
using Plots
font = Plots.font("Helvetica", 16)
pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font, size=(3000, 1500))
# pyplot()

M = 1500
N_empirical = 1000
n_samples = 4000

theta = 0.1 # enter this as a float

recs = zeros(5, N_empirical)
nrecs = zeros(5, N_empirical)
for (i, alpha) = enumerate([0.25, 0.35, 0.45, 0.5, 0.55])
  rec, _ = EmpiricalNoAtoms(N_empirical, n_samples, alpha, theta, true)
  recs[i, :] = rec
  nonrec, _ = EmpiricalNoAtoms(N_empirical, n_samples, alpha, theta, false)
  nrecs[i, :] = nonrec
end



plot(log(recs[1, :]), label="0.25")
plot!(log(recs[2, :]), label="0.35")
# plot!(log(recs[3, :]), label="0.45")
# plot!(log(recs[5, :]), label="0.55")
# plot!(log(nrecs[5, :]), label="0.55, laziest", width=4)
# xlabel!("Number of samples", fontsize=16)
# ylabel!("Log number of atoms instantiated", fontsize=16)
# savefig("atoms025-055.pdf")
