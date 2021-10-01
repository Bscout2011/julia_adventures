using LinearAlgebra

A = [
    6 1 5
    7 1 3
    7 5 9
    4 8 9
    4 1 3
]

# Problem 3
B = pinv(A)
b1 = A[2:end, :]' \ [1 0 0]'
B1 = vcat(0, b1)
B[1, :] = B1
(B * A) ≈ I(3)