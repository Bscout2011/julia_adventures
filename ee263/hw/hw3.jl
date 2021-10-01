cd("/home/alw/julia_adventures/ee263")
include("include/readclassjson.jl")

using Images
using Plots

data = readclassjson("data/tomo_data.json")

n = data["npixels"]
L = data["line_pixel_lengths"]'
y = data["y"]
x̂ = L \ y  # Least Squares solution. Backslash is overloaded to perform psuedo inverse.

img = reshape(x̂, n, n)
heatmap(img, yflip=true, aspect_ratio=:equal, color=:gist_gray, cbar=:none, framestyle=:none)


t = collect(1:1000)
z = 5*sin.(t./10 .+ 2) .+ 0.1*sin.(t) .+ 0.1*sin.(2*t .- 5)

f(k) = [k^2 k 1]

A = vcat([f(k) for k ∈ 0:-1:-9]...)
b = f(1)
y = vcat([b * (A \ z[t:-1:t-9]) for t ∈ 10:999]...)
y_interp = vcat([[b * (A^-1 * z[t:-1:t-9]) for t ∈ 10:999]...])

rms_error = sqrt((1 / 990) * sum([(z[j] - y[j-10])^2 for j ∈ 11:1000]) / ((1/990) * sum(z[11:1000].^2)))
julia_rmse = rmse(z[11:1000], y)

println(rms_error)
println(julia_rmse)

plot(t[11:20], z[11:20], label="z(t)")
plot!(t[11:20], y[1:10], label="ẑₗₛ(t)")
