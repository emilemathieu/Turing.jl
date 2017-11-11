doc"""
    PMMH(n_iters::Int, smc_alg:::SMC,)

Particle marginal Metropolis–Hastings sampler.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), MH(5,:v2))
alg = PMMH(100, SMC(20, :v1), MH(5,(:v2, (x) -> Normal(x, 1))))
```
"""
immutable PMMH <: InferenceAlgorithm
  n_iters               ::    Int               # number of iterations
  algs                  ::    Tuple             # Proposals for state & paramters
  gid                   ::    Int               # group ID
  PMMH(n_iters::Int, smc_alg::SMC, parameter_algs...) = begin
      new(n_iters, tuple(parameter_algs..., smc_alg), 0)
  end
  PMMH(alg::PMMH, new_gid) = new(alg.n_iters, alg.algs, new_gid)
end

PIMH(n_iters::Int, smc_alg::SMC) = PMMH(n_iters, smc_alg, 0)

function Sampler(alg::PMMH)
  info = Dict{Symbol, Any}()
  info[:accept_his] = []
  info[:old_likelihood_estimator] = -Inf

  n_samplers = length(alg.algs)
  samplers = Array{Sampler}(n_samplers)

  space = Set{Symbol}()

  for i in 1:n_samplers
    sub_alg = alg.algs[i]
    if isa(sub_alg, Union{SMC, MH})
      samplers[i] = Sampler(typeof(sub_alg)(sub_alg, i))
    else
      error("[PMMH] unsupport base sampling algorithm $alg")
    end
    space = union(space, sub_alg.space)
  end

  # Sanity check for space
  if !(isempty(space) && n_samplers == 1)
    @assert issubset(Turing._compiler_[:pvars], space) "[PMMH] symbols specified to samplers ($space) doesn't cover the model parameters ($(Turing._compiler_[:pvars]))"

    if Turing._compiler_[:pvars] != space
      warn("[PMMH] extra parameters specified by samplers don't exist in model: $(setdiff(space, Turing._compiler_[:pvars]))")
    end
  end

  Sampler(alg, info)
end

step(model::Function, spl::Sampler{PMMH}, vi::VarInfo, is_first::Bool) = begin
  if is_first
    spl.info[:old_prior_prob] = 0.0
  end

  smc_spl = spl.info[:smc_sampler]
  smc_spl.info[:logevidence] = []
  spl.info[:new_prior_prob] = 0.0
  spl.info[:proposal_prob] = 0.0
  spl.info[:violating_support] = false

  old_θ = copy(vi[spl])

  for local_spl in spl.info[:samplers]
    dprintln(2, "$(typeof(local_spl)) proposing...")
    runmodel(model, vi ,local_spl)
    if typeof(local_spl.alg) == MH
    #  α += local_spl.info[:new_prior_prob] - local_spl.info[:old_prior_prob] + local_spl.info[:proposal_ratio]
    elseif typeof(local_spl.alg) == SMC
      α += local_spl.info[:logevidence][end] - spl.info[:old_likelihood_estimator]
    end
  end

  # old_z = copy(vi[smc_spl])
  #
  # dprintln(2, "Propose new parameters from proposals...")
  # if !isempty(spl.alg.space)
  #   old_θ = copy(vi[spl])
  #
  #   vi = model(vi=vi, sampler=spl)
  #
  #   if spl.info[:violating_support]
  #     dprintln(2, "Early rejection, proposal is outside support...")
  #     push!(spl.info[:accept_his], false)
  #     vi[spl] = old_θ
  #     return vi
  #   end
  # end
  #
  # dprintln(2, "Propose new state with SMC...")
  # vi = step(model, smc_spl, vi)
  #
  # dprintln(2, "computing accept rate α...")
  # α = smc_spl.info[:logevidence][end] - spl.info[:old_likelihood_estimator]
  # if !isempty(spl.alg.proposals)
  #   α += spl.info[:new_prior_prob] - spl.info[:old_prior_prob] + spl.info[:proposal_prob]
  # end
  #
  # dprintln(2, "decide wether to accept...")
  # if log(rand()) < α             # accepted
  #   ## pick a particle to be retained.
  #   push!(spl.info[:accept_his], true)
  #   spl.info[:old_likelihood_estimator] = smc_spl.info[:logevidence][end]
  #   spl.info[:old_prior_prob] = spl.info[:new_prior_prob]
  # else                      # rejected
  #   push!(spl.info[:accept_his], false)
  #   if !isempty(spl.alg.space) vi[spl] = old_θ end
  #   vi[smc_spl] = old_z
  # end

  vi
end

sample(model::Function, alg::PMMH;
       save_state=false,         # flag for state saving
       resume_from=nothing,      # chain to continue
       reuse_spl_n=0             # flag for spl re-using
      ) = begin

    spl = Sampler(alg)

    # Number of samples to store
    sample_n = spl.alg.n_iters

    # Init samples
    time_total = zero(Float64)
    samples = Array{Sample}(sample_n)
    weight = 1 / sample_n
    for i = 1:sample_n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    # Init parameters
    vi = resume_from == nothing ?
              model() :
              resume_from.info[:vi]
    n = spl.alg.n_iters

    # PMMH steps
    if PROGRESS spl.info[:progress] = ProgressMeter.Progress(n, 1, "[PMMH] Sampling...", 0) end
    for i = 1:n
      dprintln(2, "PMMH stepping...")
      time_elapsed = @elapsed vi = step(model, spl, vi, i==1)

      if spl.info[:accept_his][end]     # accepted => store the new predcits
        samples[i].value = Sample(vi).value
      else                              # rejected => store the previous predcits
        samples[i] = samples[i - 1]
      end

      time_total += time_elapsed
      if PROGRESS
        haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
      end
    end

    println("[PMMH] Finished with")
    println("  Running time    = $time_total;")
    accept_rate = sum(spl.info[:accept_his]) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")

    if resume_from != nothing   # concat samples
      unshift!(samples, resume_from.value2...)
    end
    c = Chain(0, samples)       # wrap the result by Chain

    if save_state               # save state
      save!(c, spl, model, vi)
    end

    c
end
