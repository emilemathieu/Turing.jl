using Turing
using Base.Test

# srand(125)

addprocs()

@everywhere using Turing
@everywhere include_string(Main, $(read("test/ipmcmc.jl/parallel_model.jl", String)), "test/ipmcmc.jl/parallel_model.jl")

println(workers())
gibbs = IPMCMC(50, 100, 10)
chain = sample(model, gibbs)

@test_approx_eq_eps mean(chain[:z1]) 1.0 0.1
@test_approx_eq_eps mean(chain[:z2]) 1.0 0.1
@test_approx_eq_eps mean(chain[:z3]) 2.0 0.1
@test_approx_eq_eps mean(chain[:z4]) 2.0 0.1
@test_approx_eq_eps mean(chain[:mu1]) 1.0 0.1
@test_approx_eq_eps mean(chain[:mu2]) 4.0 0.1

for i in workers()
  rmprocs(i)
end
