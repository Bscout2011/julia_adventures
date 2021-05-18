### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 2ba8b802-ec51-4b15-ad8d-c4a7e85e919b
begin
	import Pkg
	Pkg.add([
		Pkg.PackageSpec(name="PlutoUI", version="0.7"), 
		Pkg.PackageSpec(name="HypertextLiteral", version="0.5"), 
		Pkg.PackageSpec(name="Distributions", version="0.25"),
		Pkg.PackageSpec(name="Plots"),
		Pkg.PackageSpec(name="Contour")
	])

	using PlutoUI
	using HypertextLiteral
	using LinearAlgebra
	using Random 
	using Distributions
	using Plots
	using Contour
end

# ╔═╡ 1eaa8b93-584b-4ea3-b1f9-40919eac516a
md"""
#### Intializing packages

_When running this notebook for the first time, this could take up to 15 minutes. Hang in there!_
"""

# ╔═╡ 1764a60f-24c5-4f12-b924-d7b984e1a7a4
Random.seed!(123);  # set the seed for reproducibility

# ╔═╡ 71b718ff-b9a3-432f-b0ce-dc9e40ff60dd
Resource("https://i.imgur.com/vf42nVH.jpeg")

# ╔═╡ f13f63ec-b509-11eb-0ea5-45a1ac8c9d4e
md"""
# Two Class Discrimination

This page illustrates how to classify between two animals, say 🐮 and 🐑, based on two attributes, such as *weight* and *fluffiness*. 

We describe the problem using Bayes' Rule

\$$
P(🐮 \; | \; x) = \frac{p(x \; | \; 🐮) P(🐮)}{p(x)}
\$$

where $x \in \mathbf R^2$ is a measurement sample of *weight* and *fluffiness*, $p(x \; | \; 🐮)$ is the likelihood distribution of each attribute given the animal is a cow, and $P(🐮)$ is the prior probability we believe the animal is a cow. The problem is: given a new sample $x$, is it more likely to be a cow or sheep? Specifically,

\$$
\text{decide 🐮 if} \quad p(x \; | 🐮) P(🐮) > p(x \; | 🐑) P(🐑), \quad \text{ otherwise, decide 🐑.}
\$$

First we assume the likelihood distributions are multivariate gaussians with a mean and covariance 

\$$
\mathcal N (\mu, \Sigma) = \frac{1}{2\pi | \Sigma |^{1/2}} \exp \left[ -.5 (x - \mu)^T \Sigma^{-1} (x - \mu) \right].
\$$

*Hint*: just think of the gaussian curve as $\sim e^{-x^2}$.

Plugging in the distribution equation into the decision rule, taking the log of both sides, and rearranging terms reveals a *quadratic* discrimination formula

\$$
g_i (x) =  -.5 (x - \mu)^T \Sigma^{-1} (x - \mu) - .5 \ln | \Sigma | + \ln P({\omega})
\$$

where $g_i, \mu, \Sigma$ are respective for each class type (🐮, 🐑). 

Below, the interactive scrubbers allow you to change the distribution parameters and see how each of the terms affect the classification. Try out three special cases.

**Case 1**: \$\Sigma 🐮 \, = \Sigma 🐑 = \sigma I$: A euclidean classifier. Class distributions are spherical. Linear decision boundary. Prior scales boundary between class means.

**Case 2**: \$\Sigma 🐮 \, = \Sigma 🐑$. Distributions are equal ellipsoids. Equal covariances give the Mahalanbois linear classifier.  Distances in feature space are scaled by the covariance principal axes.

**Case 3**: $\Sigma 🐮 \, \neq \Sigma 🐑$. Quadratic classifier. Decision boundary has many degrees of freedom.
"""

# ╔═╡ 4e8ecde0-f0a5-4eff-a858-a14b86b73dd5
let
μ_range = 0.1:.1:8
σ_range = 0.1:.1:5
cov_range = 0:.1:5
prior_range = 0.01:.1:.99
md"""
This is a "scrubbable matrix" -- click on the number and drag to change.	
	
μ 🐮 = ``(``	
 $(@bind a Scrubbable( μ_range; default=1.0))
 $(@bind b Scrubbable( μ_range; default=1.0))
``)``

μ 🐑 = 
``(``
$(@bind c Scrubbable(μ_range; default=4.0 ))
$(@bind d Scrubbable(μ_range; default=4.0))
``)``
	
Σ 🐮 = ``(``
	σ₁ = $(@bind s11 Scrubbable(σ_range; default=1)),
	σ₂ = $(@bind s12 Scrubbable(σ_range; default=1)),
	σ₁₂ = $(@bind s1cor Scrubbable(cov_range; default=0))
``)``
	
Σ 🐑 = ``(``
	σ₁ = $(@bind s21 Scrubbable(σ_range; default=1)),
	σ₂ = $(@bind s22 Scrubbable(σ_range; default=1)),
	σ₁₂ = $(@bind s2cor Scrubbable(cov_range; default=0))
``)``
	
Prior = $(@bind prior Slider(prior_range; default=.5, show_value=true))
	
Sample = ``(``	
 $(@bind s_x Scrubbable( μ_range; default=2.5))
 $(@bind s_y Scrubbable( μ_range; default=2.5))
``)``
	
**Re-run this cell to reset**
"""
end

# ╔═╡ f45de1a7-b144-4ca5-9819-3a7455a1340c
function discriminant(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix; prior=0.5)
	Σ_inv = Σ^-1
	W = -.5 * Σ_inv
	w = Σ_inv * μ
	w₀ = -.5 * transpose(μ) * Σ_inv * μ - .5 * log(det(Σ)) + log(prior)
	
	g = transpose(x) * W * x + transpose(w)*x + w₀
	g
end

# ╔═╡ d93ac03a-278d-4f70-a39c-6291e230073e
# Generate data
begin
	# discriminant(x; μ=[0., 0.], Σ=[1. 0.; 0. 1.], prior=.5) = -.5 * transpose(x - μ) * Σ₁^-1 * (x - μ) - .5 * log(det(Σ)) + log(prior)
	
	n₁ = 3000 
	n₂ = 7000
	prior₁=prior
	prior₂=1 - prior

	μ₁ = [a,b]
	μ₂ = [c,d]
	Σ₁ = [s11 s1cor
		  s1cor s12]
	Σ₂ = [s21 s2cor
		  s2cor s22]
	
	# 	Assert covariances are positive definite
	det_Σ₁ = det(Σ₁)
	det_Σ₂ = det(Σ₂)
	if (det_Σ₁ ≤ 0) || (det_Σ₂ ≤ 0)
		DomainError([det_Σ₁, det_Σ₂], "Covariance must be positive definite.")
	end
	
	# Generate multivariate gaussian data
	dist₁ = MvNormal(μ₁, Σ₁)
	dist₂ = MvNormal(μ₂, Σ₂)
	dA = rand(dist₁, n₁)
	dB = rand(dist₂, n₂)
	
	
	grid = -2:.1:10
	z₁=[pdf(dist₁, [x, y]) for y in grid, x in grid]
	z₂=[pdf(dist₂, [x, y]) for y in grid, x in grid]
	g₁ = [discriminant([x, y], μ₁, Σ₁, prior=prior₁) for y in grid, x in grid]
	g₂ = [discriminant([x, y], μ₂, Σ₂, prior=prior₂) for y in grid, x in grid]
	g = g₁ - g₂
end;

# ╔═╡ 7d8105ac-862e-48d8-9b1a-bd80387ac6e3
begin
	g_contour = lines(Contour.contour(grid, grid, g, 0))
	ys, xs = coordinates(g_contour[1])
	new_sample = [s_x, s_y]
	log_proba = discriminant(new_sample, μ₁, Σ₁, prior=prior₁) - discriminant(new_sample, μ₂, Σ₂, prior=prior₂)
	# scatter(dA[1,:], dA[2,:], 
	# 	alpha=0.1, 
	# 	xlabel="x₁", xlims=(-2, 10),
	# 	ylabel="x₂", ylims=(-2, 10),
	# 	label="ω₁")
	# scatter!(dB[1,:], dB[2,:], alpha=0.1, label="ω₂")
	Plots.contour(grid, grid, z₁, levels=3, color="blue", 
		xlabel="weight", ylabel="fluffiness", label="🐮", legend=false)
	Plots.contour!(grid, grid, z₂, levels=3, color="red", label="🐑", legend=true)
	# contour!(grid, grid, g)
	plot!(xs, ys, color="black", label="Discriminant")
	scatter!([s_x], [s_y], label="Sample")
end

# ╔═╡ 86acff24-c5cb-42f4-a1af-8da80d64f2b6
md"""
Sample classified as $(log_proba > 0 ? "🐮" : "🐑"). Log Prob: $(log_proba)
"""

# ╔═╡ 7ff8bc65-0864-4453-97de-6c88eefac295
md"""
## Reference

Some material on this website is based on "Computational Thinking, a live online Julia/Pluto textbook, https://computationalthinking.mit.edu"
"""

# ╔═╡ Cell order:
# ╟─1eaa8b93-584b-4ea3-b1f9-40919eac516a
# ╟─2ba8b802-ec51-4b15-ad8d-c4a7e85e919b
# ╟─1764a60f-24c5-4f12-b924-d7b984e1a7a4
# ╟─71b718ff-b9a3-432f-b0ce-dc9e40ff60dd
# ╟─f13f63ec-b509-11eb-0ea5-45a1ac8c9d4e
# ╟─4e8ecde0-f0a5-4eff-a858-a14b86b73dd5
# ╟─f45de1a7-b144-4ca5-9819-3a7455a1340c
# ╟─d93ac03a-278d-4f70-a39c-6291e230073e
# ╟─7d8105ac-862e-48d8-9b1a-bd80387ac6e3
# ╟─86acff24-c5cb-42f4-a1af-8da80d64f2b6
# ╟─7ff8bc65-0864-4453-97de-6c88eefac295
