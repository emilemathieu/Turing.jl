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
  s_n = std(s_nm, 1)

  return E_n, s_n

end


## tests
using Plots
# pyplot()

M = 1500
N_theory = 65
N_empirical = 1000
n_samples = 10000

alpha = 0.25
theta = 0.1 # enter this as a float

empirical_rec, s_rec = EmpiricalNoAtoms(N_empirical, n_samples, alpha, theta, true)
empirical_nonrec, s_nonrec = EmpiricalNoAtoms(N_empirical, n_samples, alpha, theta, false)

theory = TheoryExpNoAtoms(N_theory,M,alpha,theta, true)
theory_nonrec = TheoryExpNoAtoms(N_theory,M,alpha, theta, false)

println(empirical_rec)
println(empirical_nonrec)
println(theory_nonrec)
println(theory)


plot(empirical_rec', yerrors=s_rec')
plot!(empirical_nonrec', yerrors=s_nonrec')
plot!(theory')
plot!(theory_nonrec)


#ylims!((1, 8))
#plot(log.(1:M),log.(P_nm[1:M,1:50]),xlabel="m",ylabel="\ln P(M_n \leq m)",yrotation=90,legend=false)
#loglog(P_nm[1:M,1:50])
# plot(p_nm[1:M,1:60])

#p=MaxAtomProb(10,1,alpha,theta)
#println(p)
