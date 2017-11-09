"""
Notes:
 - `rand` will store randomness only when trace type matches TraceR.
 - `randc` never stores randomness. [REMOVED]
 - `randr` will store and replay randomness regardless trace type (N.B. Particle Gibbs uses `randr`).
 - `fork1` will perform replaying immediately and fix the particle weight to 1.
 - `fork2` will perform lazy replaying and accumulate likelihoods like a normal particle.
"""

module Traces
using Turing: VarInfo, Sampler, getvns, NULL, getretain

# Trick for supressing some warning messages.
#   URL: https://github.com/KristofferC/OhMyREPL.jl/issues/14#issuecomment-242886953
macro suppress_err(block)
    quote
        if ccall(:jl_generating_output, Cint, ()) == 0
            ORIGINAL_STDERR = STDERR
            err_rd, err_wr = redirect_stderr()

            value = $(esc(block))

            REDIRECTED_STDERR = STDERR
            # need to keep the return value live
            err_stream = redirect_stderr(ORIGINAL_STDERR)

            return value
        end
    end
end

include("taskcopy.jl")
include("tarray.jl")

export Trace, TraceR, TraceC, current_trace, fork, fork2, randr, TArray, tzeros,
       localcopy, @suppress_err

type Trace{T}
  task  ::  Task
  vi    ::  VarInfo
  spl   ::  Union{Void, Sampler}
  Trace() = (res = new(); res.vi = VarInfo(); res.spl = nothing; res)
end

# NOTE: this function is called by `forkr`
function (::Type{Trace{T}}){T}(f::Function)
  res = Trace{T}();
  # Task(()->f());
  res.task = Task( () -> begin res=f(); produce(Val{:done}); res; end )
  if isa(res.task.storage, Void)
    res.task.storage = ObjectIdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

function (::Type{Trace{T}}){T}(f::Function, spl::Sampler, vi :: VarInfo)
  res = Trace{T}();
  res.spl = spl
  # Task(()->f());
  res.vi = deepcopy(vi)
  res.vi.index = 0
  res.vi.num_produce = 0
  res.task = Task( () -> begin _=f(vi=res.vi, sampler=spl); produce(Val{:done}); _; end )
  if isa(res.task.storage, Void)
    res.task.storage = ObjectIdDict()
  end
  res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
  res
end

typealias TraceR Trace{:R} # Task Copy
typealias TraceC Trace{:C} # Replay

# step to the next observe statement, return log likelihood
Base.consume(t::Trace) = (t.vi.num_produce += 1; Base.consume(t.task))

# Task copying version of fork for both TraceR and TraceC.
function forkc(trace :: Trace, is_ref::Bool=false)
  newtrace = typeof(trace)()
  newtrace.task = Base.copy(trace.task)
  newtrace.spl = trace.spl

  newtrace.vi = deepcopy(trace.vi)
  if is_ref
    n_rand = min(trace.vi.index, length(getvns(trace.vi, trace.spl)))
    newtrace.vi[getretain(newtrace.vi, n_rand, trace.spl)] = NULL
  end

  newtrace.task.storage[:turing_trace] = newtrace
  newtrace
end

# fork s and replay until observation t; drop randomness between y_t:T if keep == false
#  N.B.: PG requires keeping all randomness even we only replay up to observation y_t
function forkr(trace :: TraceR, t :: Int, keep :: Bool)
  # Step 0: create new task and copy randomness
  newtrace = TraceR(trace.task.code)
  newtrace.spl = trace.spl

  newtrace.vi = deepcopy(trace.vi)
  newtrace.vi.index = 0
  newtrace.vi.num_produce = 0

  # Step 1: Call consume t times to replay randomness
  map(i -> consume(newtrace), 1:t)

  # Step 2: Remove remaining randomness if keep==false
  if !keep
    index = newtrace.vi.index
    newtrace.vi[getretain(newtrace.vi, index, trace.spl)] = NULL
  end

  newtrace
end

# Default fork implementation, replay immediately.
fork(s :: TraceR) = forkr(s, s.vi.num_produce, false)
fork(s :: TraceC) = forkc(s)

# Lazily replay on demand, note that:
#  - lazy replay is only possible for TraceR
#  - lazy replay accumulates likelihoods
#  - lazy replay is useful for implementing PG (i.e. ref particle)
fork2(s :: TraceR) = forkr(s, 0, true)

current_trace() = current_task().storage[:turing_trace]

end
