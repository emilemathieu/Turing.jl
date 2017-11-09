# Test TraceC and TraceR

using Turing
using Distributions
using Gallium
import Turing: Trace, TraceR, TraceC, current_trace, fork, fork2, VarName, Sampler
import Turing: acclogp!,getlogp,setlogp!,dualpart,getvns,getrange,CHUNKSIZE,getval,realpart,SEEDS,runmodel,VarInfo,gradient

global n = 0

@model NLSSM1(y) = begin
  N = length(y)
  sigma2_v = sigma2_w = 10
  # sigma2_v= 10; sigma2_w = 1

  x = Vector{Any}(N)
  x[1] ~ Normal(0, sqrt(5))
  y[1] ~ Normal(x[1]^2/20, sqrt(sigma2_w))
  for i in 2:N
    x[i] ~ Normal(x[i-1]/2 + 25*x[i-1]/(1+x[i-1]^2) + 8*cos(1.2*i), sqrt(sigma2_v))
    y[i] ~ Normal(x[i]^2/20, sqrt(sigma2_w))
  end
end

@model gibbstest(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

model1 = gibbstest([0.1,0.2])
model2 = NLSSM1([1.0,2.2,1.5])

alg = PDMCMC(1, 0.1)
spl = Turing.Sampler(alg)
dist = Normal(0, 1)

gradient3(model, vi, spl=nothing) = begin
θ_hash = hash(vi[spl])
# Initialisation
grad = Vector{Float64}()

# Split keys(vi) into chunks,
dprintln(4, "making chunks...")
vn_chunk = Set{VarName}(); vn_chunks = []; chunk_dim = 0;

vns = getvns(vi, spl); vn_num = length(vns)

for i = 1:vn_num
  l = length(getrange(vi, vns[i]))           # dimension for the current variable
  if chunk_dim + l > CHUNKSIZE
    push!(vn_chunks,        # store the previous chunk
          (vn_chunk, chunk_dim))
    vn_chunk = []           # initialise a new chunk
    chunk_dim = 0           # reset dimension counter
  end
  push!(vn_chunk, vns[i])       # put the current variable into the current chunk
  chunk_dim += l            # update dimension counter
end
push!(vn_chunks,            # push the last chunk
      (vn_chunk, chunk_dim))

# Chunk-wise forward AD
for (vn_chunk, chunk_dim) in vn_chunks
  # 1. Set dual part correspondingly
  dprintln(4, "set dual...")
  dim_count = 1
  for i = 1:vn_num
    range = getrange(vi, vns[i])
    l = length(range)
    vals = getval(vi, vns[i])
    if vns[i] in vn_chunk        # for each variable to compute gradient in this round
      for i = 1:l
        vi[range[i]] = ForwardDiff.Dual{CHUNKSIZE, Float64}(realpart(vals[i]), SEEDS[dim_count])
        dim_count += 1      # count
      end
    else                    # for other varilables (no gradient in this round)
      for i = 1:l
        vi[range[i]] = ForwardDiff.Dual{CHUNKSIZE, Float64}(realpart(vals[i]))
      end
    end
  end
  dprintln(4, "set dual done")

  # 2. Run model
  dprintln(4, "run model...")
  println("run model...")
  vi.index = 0
  vi.logp = Vector{Real}()
  # vi = runmodel(model, vi, spl)
  if spl != nothing spl.info[:total_eval_num] += 1 end
  vi = model(vi=vi, sampler=spl) # run model

  println("vi.logp")
  println(vi.logp)
  println("dualpart(-vi.logp)")
  println(dualpart(-vi.logp))
  println("collect(dualpart(-vi.logp))")
  println(collect(dualpart(-vi.logp)))
  println(typeof(collect(dualpart(-vi.logp))))
  # println(collect(dualpart(-vi.logp))[1:chunk_dim])
  # 3. Collect gradient
  dprintln(4, "collect gradients from logp...")
  # append!(grad, collect(dualpart(-getlogp(vi)))[1:chunk_dim])
  # append!(grad, collect(dualpart(-getlogp(vi)))[1:chunk_dim])
end

grad
end

gradient4(t, spl=nothing) = begin
θ_hash = hash(t.vi[spl])
# Initialisation
grad = Vector{Float64}()

# Split keys(t.vi) into chunks,
dprintln(4, "making chunks...")
vn_chunk = Set{VarName}(); vn_chunks = []; chunk_dim = 0;

vns = getvns(t.vi, spl); vn_num = length(vns)

for i = 1:vn_num
  l = length(getrange(t.vi, vns[i]))           # dimension for the current variable
  if chunk_dim + l > CHUNKSIZE
    push!(vn_chunks,        # store the pret.vious chunk
          (vn_chunk, chunk_dim))
    vn_chunk = []           # initialise a new chunk
    chunk_dim = 0           # reset dimension counter
  end
  push!(vn_chunk, vns[i])       # put the current variable into the current chunk
  chunk_dim += l            # update dimension counter
end
push!(vn_chunks,            # push the last chunk
      (vn_chunk, chunk_dim))

# Chunk-wise forward AD
for (vn_chunk, chunk_dim) in vn_chunks
  # 1. Set dual part correspondingly
  dprintln(4, "set dual...")
  dim_count = 1
  for i = 1:vn_num
    range = getrange(t.vi, vns[i])
    l = length(range)
    vals = getval(t.vi, vns[i])
    if vns[i] in vn_chunk        # for each variable to compute gradient in this round
      for i = 1:l
        t.vi[range[i]] = ForwardDiff.Dual{CHUNKSIZE, Float64}(realpart(vals[i]), SEEDS[dim_count])
        dim_count += 1      # count
      end
    else                    # for other varilables (no gradient in this round)
      for i = 1:l
        t.vi[range[i]] = ForwardDiff.Dual{CHUNKSIZE, Float64}(realpart(vals[i]))
      end
    end
  end
  dprintln(4, "set dual done")

  # 2. Run model
  dprintln(4, "run model...")
  # vi = runmodel(model, vi, spl)
  setlogp!(vi, zero(Real))
  res = consume(t)
  println(res)
  println(t.vi)
  println(getlogp(t.vi))
  # 3. Collect gradient
  dprintln(4, "collect gradients from logp...")
  append!(grad, collect(dualpart(-getlogp(t.vi)))[1:chunk_dim])
end

grad
end

# spl.info[:total_eval_num] = 0
# vi = model2()
# t = TraceC(model2, spl, vi)
# setlogp!(t.vi, zero(Real))
# gradient4(t, spl)

vi = model2()
vi = runmodel(model2, vi, spl)
# vi.logp = Vector{Real}()
grad = gradient3(model2, vi, spl)

# @step gradient3(vi, spl)
