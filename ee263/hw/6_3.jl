include("/home/alw/julia_adventures/ee263/include/readclassjson.jl")

using Plots
using LinearAlgebra
using Statistics

data = readclassjson("data/bs_det_data.json")
Y = data["Y"]
T = data["T"]
s = data["s"]

u, σ, v = svd(Y')

σ₁ = s[1]
w = (sqrt(T) / σ₁) * v[:,1]
ŝ = (w' * Y)'

preds = sign.(ŝ)
error = mean(preds .!= s)
println("Error:", error)

histogram(ŝ, bins=60)

