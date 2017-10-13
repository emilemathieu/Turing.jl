function ρ(x, k, v=var(x))
    x1 = @view(x[1:(end-k)])
    x2 = @view(x[(1+k):end])
    V = sum((x1 .- x2).^2)/length(x1)
    1-V/(2*v)
end

"""
Factor `τ` for effective sample size and the last lag which was used
in the estimation.
"""
function ess_factor(x)
    v = var(x)
    invfactor = 1 + 2*ρ(x, 1, v)
    K = 2
    while K < length(x)-2
        increment = ρ(x, K, v) + ρ(x, K+1, v)
        if increment < 0
            break
        else
            K += 2
            invfactor += 2*increment
        end
    end
    length(x)/invfactor#, K #1/invfactor, K
end

f = open("test/pmmh.jl/output.csv")
lines = readlines(f)
N = size(lines)[1]
res = zeros(N, 2)

for l in 1:N
  line = lines[l]
   m,s = split(line, ",")
   res[l, 1] = parse(split(m, " ")[2][1:end-1])
   res[l, 2] = parse(split(s, " ")[2][1:end-2])
end
println(ess_factor(res[:,1])/N)
println(ess_factor(res[:,2])/N)
