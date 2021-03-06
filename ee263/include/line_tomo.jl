"""
Given an `n`×`n` square grid, and a straight line over that grid `d` units from
the center of the grid (measured in grid lengths orthogonally to the line) at an
angle of `θ` radians from the x-axis, compute length of the line passing through
square in the grid. Return an `n×n` array, whose each element is the length of
the line over the respective grid square.
"""
function line_pixel_length(d::Float64, θ::Float64, n::Int)

    # for angles in [π/4, 3π/4], flip along diagonal and call recursively
    if π/4 < θ < 3π/4
        L = line_pixel_length(d, π/2-θ, n)
        return collect(transpose(L))
    end

    # for angle in [3π/4, π], redefine line to go in opposite direction
    if θ > π/2
        d = -d
        θ = θ - π
    end

    # for angle in [-π/4, 0], flip along x-axis and call recursively
    if θ < 0
        L = line_pixel_length(-d, -θ, n)
        return reverse(L, dims=1)
    end

    if !(0 ≤ θ ≤ π/2)
        error("invalid angle")
    end

    L = zeros(n, n)
    cosθ = cos(θ)
    sinθ = sin(θ)
    tanθ = sinθ / cosθ

    x0 = n/2 - d * sinθ
    y0 = n/2 + d * cosθ

    y = y0 - x0 .* tanθ
    jy = ceil(Int, y)
    dy = rem(y + n, 1)

    for jx in 1:n
        dynext = dy + tanθ
        if dynext < 1
            if 1 ≤ jy ≤ n
                L[n+1-jy, jx] = 1 / cosθ
            end
            dy = dynext
        else
            if 1 ≤ jy ≤ n
                L[n+1-jy, jx] = (1 - dy) / sinθ
            end
            if 1 ≤ jy + 1 ≤ n
                L[n+1-(jy+1), jx] = (dynext - 1) / sinθ
            end
            dy = dynext - 1
            jy += 1
        end
    end

    return L
end