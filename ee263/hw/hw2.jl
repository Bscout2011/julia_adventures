include("include/readclassjson.jl")
using LinearAlgebra

data = readclassjson("data/color_perception_data.json")

r = data["R_phosphor"]
g = data["G_phosphor"]
b = data["B_phosphor"]

wavelength = data["wavelength"]

l = data["L_coefficients"]
m = data["M_coefficients"]
s = data["S_coefficients"]

test = data["test_light"]

P = [r g b]
A = [l m s]'

B = A*P

weights = B^-1 * A * test

tungsten = data["tungsten"]
sunlight = data["sunlight"]

n_A = nullspace(A)
n = n_A[:,1]
n = n .* 10 ./ tungsten

r1 = rand(20)
r2 = r1 .- n


tungsten_1 = A * (tungsten .* r1)
tungsten_2 = A * (tungsten .* r2)

print("Light 1 match: ", tungsten_1 ≈ tungsten_2)

sunlight_1 = A * (sunlight .* r1)
sunlight_2 = A * (sunlight .* r2)

print("\nLight 2 match: ", sunlight_1 ≈ sunlight_2, "\n")

data2 = readclassjson("data/one_bad_sensor.json")

A = data2["A"]
ỹ = data2["ytilde"]

m = length(ỹ)
for i ∈ 1:m
    if rank(A) == rank([A ỹ][1:m .!= i, :])
        println("Sensor ", i, " is faulty.")
    end
end
