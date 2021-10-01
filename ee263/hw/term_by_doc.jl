include("/home/alw/julia_adventures/ee263/include/readclassjson.jl")

using Plots
using LinearAlgebra
using Statistics

data = readclassjson("data/term_by_doc.json")
term = data["term"]
doc = data["document"]
A = data["A"]
m = data["m"]
n = data["n"]

A_norm = norm.(eachcol(A))
Ã = hcat([A[:,i] ./ A_norm[i] for i ∈ 1:n]...)
U, S, V = svd(Ã)

q = zeros(m)
q[53] = 1

c = Ã' * q
p = sortperm(c, rev=true)
print(p[1:5])

# plot(S, line=:stem, marker=:circle)

r = 32
W = diag(S[1:r]) * V[:,1:r]
Â = U[:,1:r] * 
for i ∈ 1:32

end