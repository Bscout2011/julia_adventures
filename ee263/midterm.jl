using LinearAlgebra
using Plots

include("include/readclassjson.jl")
include("include/line_tomo.jl")

function line(θ, d)
    a = zeros(n)
    for p = 1:n
        i = (p-1) ÷ w + 1
        j = (p-1) % w + 1
        a[p] = abs((j-w/2)*sin(θ) - (i-h/2)*cos(θ) - d) < 2
    end

    return a
end

data = readclassjson("data/tomodata.json")

w = data["w"]
h = data["h"]  
L = data["L"]  # L[:, 1] is θ, L[:, 2] = d
y = data["y"]  # log-absorbtion

n = w * h
m = size(L)[1]
μ = 0.1
k = [1 10 50 100 200 500 1000 2000]


plot_layout = []

i = 100
θ = L[i,1]
d = L[i,2]
A = line.(L[1:2000,1], L[1:2000,2])
A = hcat(A...)'

for i ∈ k
    x̂ = (transpose(A[1:i,:]) * A[1:i,:] + μ * I(n))^-1 * transpose(A[1:i,:])*y[1:i]
    img = reshape(x̂, w, h)'
    my_plot = heatmap(img, yflip=true, color=:viridis, cbar=:none, framestyle=:none)
    push!(plot_layout, my_plot)
end


plot(plot_layout..., layout=(2,4), legend=false)
