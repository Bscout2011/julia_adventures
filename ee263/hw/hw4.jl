include("include/readclassjson.jl")
using Statistics
using Plots
using LinearAlgebra

data = readclassjson("data/inductor_data.json")
curve = readclassjson("data/curve_smoothing.json")

d = data["d"]
w = data["w"]
N = data["N"]
D = data["D"]
L = data["L"]
n = data["n"]

A = [ones(N) log.(n) log.(w) log.(d) log.(D)]
ls_sol = A \ log.(L)

α = exp(ls_sol[1])
L̂ = [α * n[i]^ls_sol[2] * w[i]^ls_sol[3] * d[i]^ls_sol[4] * D[i]^ls_sol[5] for i ∈ 1:N]

error = 100 * abs.(L - L̂) ./ L

println("Error: ", mean(error))

f = curve["f"]
n = length(f)

p = plot(1:n, f, style=:dash, label="f")

mu_vals = [0 1e-8 1e-6 1e-4 100]
J1_vals = zeros(0)
J2_vals = zeros(0)

for μ ∈ [0 1e-8 1e-6 1e-4 100]
    A = I(n) / sqrt(n)
    S = zeros(n-2, n)
    for i in 2:n-1
        S[i-1, i-1] = 1
        S[i-1, i] = -2
        S[i-1, i+1] = 1
    end
    S = (n^2 / sqrt(n-2)) * S
    F = sqrt(μ) * S

    Ã = vcat(A, F)
    ỹ = vcat(f./sqrt(n), zeros(n-2))
    g = Ã \ ỹ

    plot!(p, 1:n, g, label="μ = $μ")

    J1 = norm(A * g - (1/sqrt(n))*f)
    J2 = norm(S * g)
    append!(J1_vals, J1)
    append!(J2_vals, J2)
end

p

plot(J1_vals, J2_vals, 
label="Optimal Tradeoff Curve",
xlabel="Sum-square deviation",
ylabel="Sum-square Curvature")