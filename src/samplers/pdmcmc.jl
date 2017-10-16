doc"""
    PDMCMC(n_iters::Int, epsilon::Float64, tau::Int)

Hamiltonian Monte Carlo sampler.

Usage:

```julia
PDMCMC(1000, 0.05, 10)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.5, 2]), PDMCMC(1000, 0.05, 10))
```
"""
immutable PDMCMC <: InferenceAlgorithm
  n_iters   ::  Int       # number of samples
  epsilon   ::  Float64   # time discretization
  space     ::  Set       # sampling space, emtpy means all
  gid       ::  Int       # group ID
  PDMCMC(n_iters::Int, epsilon::Float64) = new(n_iters, epsilon, Set(), 0)
  PDMCMC(n_iters::Int, epsilon::Float64, space...) =
    new(n_iters, epsilon, isa(space, Symbol) ? Set([space]) : Set(space), 0)
  PDMCMC(alg::PDMCMC, new_gid::Int) = new(alg.n_iters, alg.epsilon, alg.space, new_gid)
end

Sampler(alg::PDMCMC) = begin
  info=Dict{Symbol, Any}()

  # For caching gradient
  info[:grad_cache] = Dict{UInt64,Vector}()
  info[:total_eval_num] = 0
  info[:m] = 0

  Sampler(alg, info)
end

step(model::Function, spl::Sampler{PDMCMC}, vi::VarInfo, is_first::Bool) = begin

m = spl.info[:m]
x = copy(vi[spl])

if is_first
  spl.info[:v] = rand(MultivariateNormal(ones(length(x))))
end
v = spl.info[:v]

### 1. Sample u
u = zeros(m)
# compute gamma[i]s
gamma = copy(-vi.logp)

for i in 1:m
  u[i] = rand(Uniform(0, gamma[i]))
end

### 2. Constraints violated ?

# Simulate ODE # NOTE: currently only linear dynamics, later HMC
# x_sim, v = linear_dynamics(x, v, spl.alg.epsilon)
x_sim, v = x + spl.alg.epsilon * v, v

violated = falses(m)
for i in 1:m
 if u[i] >= gamma[i] violated[i] = true end
end
V = find(violated .== true)

if isempty(V)
  x = x_sim

### 3.
else
# compute local gradients not respecting the constraints
  #local_grad = local_gradients(model, vi, spl)
  #grad_u_bar = sum(local_grad .* violated, 1)'

  # v_bounce = reflect_bps(grad_u_bar, v*epsilon)
  # x_bounce = x_bounce - v_bounce
  # x_bounce, v_bounce = linear_dynamics(x_bounce, v_bounce, spl.alg.epsilon)
  # violated_bounce = falses(m)
  # vi_bounce = deepcopy(vi)
  # vi_bounce[spl] = x_bounce
  # vi_bounce = runmodel_pdmcmc(model, vi_bounce, spl)
  # gamma = copy(-vi_bounce.logp)
  # for i in 1:m
  #  if u[i] >= gamma[i] violated[i] = true end
  # end
  # V_bounce = find(violated_bounce .== true)

  # if V_bounce == V_bounce
    # v = reflect_bps(grad_u_bar, v)
  # else
  x, v = x, -v
  # end
end

vi[spl] = copy(x)
spl.info[:v] = copy(v)

vi
end

function sample(model::Function, alg::PDMCMC;
                                chunk_size=CHUNKSIZE,     # set temporary chunk size
                                save_state=false,         # flag for state saving
                                resume_from=nothing,      # chain to continue
                                reuse_spl_n=0,            # flag for spl re-using
                               )

  default_chunk_size = CHUNKSIZE  # record global chunk size
  setchunksize(chunk_size)        # set temp chunk size

  spl = reuse_spl_n > 0 ?
        resume_from.info[:spl] :
        Sampler(alg)

  # Initialization
  time_total = zero(Float64)
  n = reuse_spl_n > 0 ?
      reuse_spl_n :
      alg.n_iters
  samples = Array{Sample}(n)
  weight = 1 / n
  for i = 1:n
    samples[i] = Sample(weight, Dict{Symbol, Any}())
  end

  vi = resume_from == nothing ?
       model() :
       deepcopy(resume_from.info[:vi])

  if spl.alg.gid == 0
    link!(vi, spl)
    runmodel_pdmcmc(model, vi, spl)
  end

  if PROGRESS spl.info[:progress] = ProgressMeter.Progress(n, 1, "[PDMCMC] Sampling...", 0) end
  for i = 1:n
    dprintln(2, "PDMCMC stepping...")

    # time_elapsed = @elapsed vi = step(model, spl, vi, i == 1)
    time_elapsed = 0.0
    vi = step(model, spl, vi, i == 1)
    time_total += time_elapsed

    samples[i].value = Sample(vi, spl).value

    if PROGRESS ProgressMeter.next!(spl.info[:progress]) end
  end

  println("[PDMCMC] Finished with")
  println("  Running time        = $time_total;")

  setchunksize(default_chunk_size)      # revert global chunk size

  if resume_from != nothing   # concat samples
    unshift!(samples, resume_from.value2...)
  end
  c = Chain(0, samples)       # wrap the result by Chain
  if save_state               # save state
    # Convert vi back to X if vi is required to be saved
    if spl.alg.gid == 0 invlink!(vi, spl) end
    save!(c, spl, model, vi)
  end

  c
end

assume(spl::Sampler{PDMCMC}, dist::Distribution, vn::VarName, vi::VarInfo) = begin
  dprintln(2, "assuming...")
  updategid!(vi, vn, spl)
  r = vi[vn]
  println("assume: ", logpdf_with_trans(dist, r, istrans(vi, vn)))
  push!(vi.logp, logpdf_with_trans(dist, r, istrans(vi, vn)))
  spl.info[:m] += 1
  # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
  r
end

observe(spl::Sampler{PDMCMC}, d::Distribution, value::Any, vi::VarInfo) = begin
  # acclogp!(vi, logpdf(d, value))
  println("assume: ", logpdf(d, value))
  push!(vi.logp, logpdf(d, value))
  spl.info[:m] += 1
end
