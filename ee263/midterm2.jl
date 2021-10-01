using Elliptic
using LinearAlgebra
using Plots

logrange(x1, x2, n) = (10^y for y in range(log10(x1), log10(x2), length=n))

r_min = 0.3
r_max = 0.4
L = 1.02

m = 1000
d = 0.225
θ = [π*(i-1) / (m-1) for i = 1:m]
z = d * cos.(θ)
r = d * sin.(θ)

n_r = 5
n_z = 20
n = n_r * n_z
z̃ = [-L/2 + L*(i-1)/(n_z-1) for i = 1:n_z]
a = [r_min + (r_max-r_min)*(i-1)/(n_r - 1) for i = 1:n_r]

# iterate through axial, then radial loops.
A = zeros(m, n_r*n_z)
for i = 1:m
    for j_z = 1:n_z
        for j_r = 1:n_r
            q = (a[j_r] + r[i])^2 + (z[i] - z̃[j_z])^2
            k = sqrt(4 * a[j_r] * r[i] / q)
            K, E = ellipke(k*k)
            coef = 1 / (2 * π * √q) * ((a[j_r]^2 - r[i]^2 - (z[i] - z̃[j_z])^2) * E / ((a[j_r] - r[i])^2 + (z[i] - z̃[j_z])^2) + K)
            j = (j_z - 1) * n_r + j_r
            A[i, j] = coef
        end
    end
end

μ_min = 0
F_min = Inf
x̂_min = zeros(n)
P_max = 0.025

J_pow = []
J_accuracy = []
for μ in logrange(.01, 100, 100)
    Ã = vcat(A, √μ * I(n)) 
    b = vcat(ones(m), zeros(n))

    x̂ = Ã \ b
    push!(J_pow, norm(x̂))
    acc = norm(A * x̂ - ones(m))
    push!(J_accuracy, acc)
    # println(μ, maximum(x̂))

    if all(x̂ .< P_max) && acc < F_min   
        μ_min = μ
        F_min = acc
        x̂_min = x̂
    end
end

plot(J_accuracy, J_pow, xlabel="Power", ylabel="Accuracy")

heatmap(reshape(x̂_min, n_z, n_r), title="μ = $μ_min")

y = A * x̂_min
plot(θ, y, xlabel="θ", ylabel="Magnetic Field")