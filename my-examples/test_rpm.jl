using Turing

# P1 = rand(NGGP(0.5, 1.0, Normal()))
P1 = rand(NGGP(0.5, 1.0, Normal(), true))
# P1 = rand(PYP(0.5, 1.0, Normal()))
# P1 = rand(PYP(0.5, 1.0, Normal(), true))

K = 1000
N = 100
size1 = zeros(K); size2 = zeros(K)

for k in 1:K
println(k)
res1 = zeros(N); res2 = zeros(N)
for i in 1:N
  # println(i)
  res1[i] = rand(P1)
  # println(i)
  # res2[i] = rand(P2)
end
size1[k] = length(unique(res1))
# size2[k] = length(unique(res2))
end

println("Mean nb cluster 1 ", mean(size1))
# println("Mean nb cluster 2", mean(size2))
