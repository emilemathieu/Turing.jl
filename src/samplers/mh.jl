doc"""
    HMC(n_iters::Int)

Metropolis-Hastings sampler.

Usage:

```julia
MH(100, (:m, (x) -> Normal(x, 0.1)))
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

# sample(gdemo([1.5, 2]), MH(1000, (:m, (x) -> Normal(x, 0.1)), :s)))
```
"""
immutable MH <: InferenceAlgorithm
  n_iters   ::  Int       # number of iterations
  space     ::  Set       # sampling space, emtpy means all
  gid       ::  Int       # group ID
  function MH(n_iters::Int, space...)
    new_space = Set()
    proposals = Dict{Symbol,Any}()
    for element in space
        if isa(element, Symbol)
          push!(new_space, element)
        else
          @assert isa(element[1], Symbol) "[MH] ($element[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, (x) -> Normal(x, 0.1)))"
          push!(new_space, element[1])
          proposals[element[1]] = element[2]
        end
    end

    @assert !isempty(new_space) "[MH] (If no parameter is specified, use only SMC)"
    new_space = Set(new_space)
    new(n_iters, smc_alg, proposals, new_space, 0)
  end
end

Sampler(alg::MH) = begin
    info = Dict{Symbol, Any}()
    info[:accept_his] = []

    # Sanity check for space
    space = union(alg.space, alg.smc_alg.space)
    @assert issubset(Turing._compiler_[:pvars], space) "[MH] symbols specified to samplers ($space) doesn't cover the model parameters ($(Turing._compiler_[:pvars]))"

    if Turing._compiler_[:pvars] != space
    warn("[MH] extra parameters specified by samplers don't exist in model: $(setdiff(space, Turing._compiler_[:pvars]))")
    end

    Sampler(alg, info)
end

step(model::Function, spl::Sampler{MH}, vi::VarInfo, is_first::Bool) = begin
  if is_first
    spl.info[:old_likelihood_estimator] = -Inf
    spl.info[:old_prior_prob] = 0.0
    spl.info[:accept_his] = []
  end

  new_likelihood_estimator = 0.0
  spl.info[:new_prior_prob] = 0.0
  spl.info[:proposal_prob] = 0.0
  spl.info[:violating_support] = false

  old_θ = copy(vi[spl])

  dprintln(2, "Propose new parameters from proposals...")
  vi = model(vi=vi, sampler=spl)

  if spl.info[:violating_support]
    dprintln(2, "Early rejection, proposal is outside support...")
    push!(spl.info[:accept_his], false)
    vi[spl] = old_θ
    return vi
  end

  new_likelihood_estimator = getlogp(vi)
  dprintln(2, "computing accept rate α...")
  α = new_likelihood_estimator - spl.info[:old_likelihood_estimator]
  if !isempty(spl.alg.proposals)
    α += spl.info[:new_prior_prob] - spl.info[:old_prior_prob] + spl.info[:proposal_prob]
  end

  dprintln(2, "decide wether to accept...")
  if log(rand()) < α             # accepted
    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    vi = particles[indx].vi

    push!(spl.info[:accept_his], true)
    spl.info[:old_likelihood_estimator] = new_likelihood_estimator
    spl.info[:old_prior_prob] = spl.info[:new_prior_prob]
  else                      # rejected
    push!(spl.info[:accept_his], false)
    vi[spl] = old_θ
  end

  vi
end

function sample{T<:Hamiltonian}(model::Function, alg::T;
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

  alg_str = "MH"

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
    # link!(vi, spl)
    # runmodel(model, vi, spl)
    vi.index = 0; setlogp!(vi, zero(Real)); vi = model(vi=vi, sampler=spl)
  end

  # MH steps
  if PROGRESS spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0) end
  for i = 1:n
    dprintln(2, "$alg_str stepping...")

    time_elapsed = @elapsed vi = step(model, spl, vi, i == 1)
    time_total += time_elapsed

    if spl.info[:accept_his][end]     # accepted => store the new predcits
      samples[i].value = Sample(vi, spl).value
    else                              # rejected => store the previous predcits
      samples[i] = samples[i - 1]
    end
    samples[i].value[:elapsed] = time_elapsed

    if PROGRESS ProgressMeter.next!(spl.info[:progress]) end
  end

  println("[$alg_str] Finished with")
  println("  Running time        = $time_total;")
  accept_rate = sum(spl.info[:accept_his]) / n  # calculate the accept rate
  println("  Accept rate         = $accept_rate;")

  setchunksize(default_chunk_size)      # revert global chunk size

  if resume_from != nothing   # concat samples
    unshift!(samples, resume_from.value2...)
  end
  c = Chain(0, samples)       # wrap the result by Chain
  if save_state               # save state
    # Convert vi back to X if vi is required to be saved
    # if spl.alg.gid == 0 invlink!(vi, spl) end
    save!(c, spl, model, vi)
  end

  c
end

function rand_truncated(dist, lowerbound, upperbound)
    notvalid = true
    x = 0.0
    while (notvalid)
        x = rand(dist)
        notvalid = ((x < lowerbound) | (x > upperbound))
    end
    return x
end

assume(spl::Sampler{MH}, dist::Distribution, vn::VarName, vi::VarInfo) = begin
    if vn.sym in spl.alg.space
      vi.index += 1
      if ~haskey(vi, vn) #NOTE: When would that happens ??
        r = rand(dist)
        push!(vi, vn, r, dist, spl.alg.gid)
        spl.info[:cache_updated] = CACHERESET # sanity flag mask for getidcs and getranges
      elseif vn.sym in keys(spl.alg.proposals) # Custom proposal for this parameter
        oldval = getval(vi, vn)[1]
        proposal = spl.alg.proposals[vn.sym](oldval)
        if typeof(proposal) == Distributions.Normal{Float64} # If Gaussian proposal
          σ = std(proposal)
          lb = support(dist).lb
          ub = support(dist).ub
          stdG = Normal()
          r = rand_truncated(proposal, lb, ub)
          # cf http://fsaad.scripts.mit.edu/randomseed/metropolis-hastings-sampling-with-gaussian-drift-proposal-on-bounded-support/
          spl.info[:proposal_prob] += log(cdf(stdG, (ub-oldval)/σ) - cdf(stdG,(lb-oldval)/σ))
          spl.info[:proposal_prob] -= log(cdf(stdG, (ub-r)/σ) - cdf(stdG,(lb-r)/σ))
      else # Other than Gaussian proposal
          r = rand(proposal)
          if (r < support(dist).lb) | (r > support(dist).ub) # check if value lies in support
            spl.info[:violating_support] = true
            r = oldval
          end
          spl.info[:proposal_prob] -= logpdf(proposal, r) # accumulate pdf of proposal
          reverse_proposal = spl.alg.proposals[vn.sym](r)
          spl.info[:proposal_prob] += logpdf(reverse_proposal, oldval)
        end
        spl.info[:new_prior_prob] += logpdf(dist, r) # accumulate pdf of prior
      else # Prior as proposal
        r = rand(dist)
      end
      setval!(vi, vectorize(dist, r), vn)
      setgid!(vi, spl.alg.gid, vn)
      r
    else
      vi[vn]
    end
end

# assume{D<:Distribution}(spl::Sampler{MH}, dists::Vector{D}, vn::VarName, var::Any, vi::VarInfo) =
#   error("[Turing] MH doesn't support vectorizing assume statement")
#
# observe(spl::Sampler{MH}, d::Distribution, value::Any, vi::VarInfo) =
#   observe(nothing, d, value, vi)
#
# observe{D<:Distribution}(spl::Sampler{MH}, ds::Vector{D}, value::Any, vi::VarInfo) =
#   observe(nothing, ds, value, vi)
