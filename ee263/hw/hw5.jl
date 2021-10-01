include("../include/readclassjson.jl")
using Statistics
using Plots
using LinearAlgebra

data = readclassjson("ee263/data/gauss_fit_data.json")

N = data["N"]
t = data["t"]
y = data["y"]


function fit_gaussian(p_init)
    p = p_init
    E = Float64[]
    for i = 1:100
        a, mu, sigma = p
        w = exp.(-(t .- mu).^2 / sigma^2)
        A =[w 2*a*(t.-mu)/sigma^2 .* w 2*a*(t.-mu).^2/sigma^3 .* w]
        f = a .* w
        b = y - f
        Deltap = A \ b
        p += Deltap
        push!(E, sqrt(sum((f - y).^2) / N))
    end
    p..., E
end

fit_gaussian(a_init, mu_init, sigma_init) = fit_gaussian([a_init, mu_init, sigma_init])

a, mu, sigma, E = fit_gaussian(100, 40, 1)

f = a * exp.(-(t .- mu).^2 / sigma^2)

gr(size=(1200, 1000))
scatter(t, y, label="data", marker=:+)
plot!(t, f, label="fitted")
# plot(E, label="rms error")