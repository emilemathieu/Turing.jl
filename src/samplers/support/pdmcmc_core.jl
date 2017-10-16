runmodel_pdmcmc(model::Function, vi::VarInfo, spl::Sampler) = begin
  dprintln(4, "run model...")
  vi.index = 0
  vi.logp = Vector{Real}()
  if spl != nothing spl.info[:total_eval_num] += 1 end
  model(vi=vi, sampler=spl) # run model
end

function reflect_bps{T<:Vector{Float64}}(n::T, v::T)::T
    v -= 2.0dot(n, v)*n/dot(n,n)
    v
end

function linear_dynamics{T<:Vector{Real}}(x::T, v::T, epsilon::Float64)::Tuple{T, T}
  x + epsilon * v, v
end

# local_gradients(model::Function, vi::VarInfo, spl::Sampler) = begin
#
# end
