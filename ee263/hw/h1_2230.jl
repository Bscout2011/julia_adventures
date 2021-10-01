### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 58cbddf4-b723-11eb-03c1-df1b4cec59a4
begin
	using Plots
	include("include/readclassjson.jl")
	
end

# ╔═╡ f9bcde53-1070-4c91-b860-dbeef1f7a9c2
begin
	data = readclassjson("pop_dyn_data.json")
	A = data["A"]
	n = data["n"]
end;

# ╔═╡ e0e01781-4c2a-4461-8cb1-d5ff16d9350e
md"""
### 2.230 Population Dynamics

An ecosystem consists of $n$ species that interact (say, 🐺 eats 🐰, 🐮 eats grass, 🐉 burns down ecosystem, and so on). We let $x(t) \in \mathbf R^n$ be a vector of deviations from the species' population equilibrium, in time $t$. In this model, time takes on discrete values $t = 0, 1, 2, \ldots$. Thus $x_3(4) < 0$ means that the population of species 3 (like a 🐄) in time period 4 is below its equilibrium value. The population (deviations) follows a discrete-time linear dynamical system

\$$
x(t+1) = Ax(t).
\$$

We refer to $x(0)$ as the *initial population perturbation*.

a) Suppose the initial population $x(0) = e_4$. What's the shortest time period where the other species populations are affected?
"""

# ╔═╡ 40e1dc49-626b-4780-9467-f7da180168de
begin
	s = -ones(Int64, n)  # this is the time period when the species' population is affected
	x = zeros(n)  # population deviation vector
	x[4] = 1  
	
	for t = 0:10
		s[(x .!= 0) .& (s .== -1)] .= t
		if all(s .!= -1)
			break
		end
		x = A * x
	end
	s
end

# ╔═╡ c218188c-1973-4383-b60f-2d22d97fe1e3
md"""
b) *Population control*. We can choose any initial perturbation that satisfies $|x_i(0)| \leq 1$ for each $i = 1, \ldots 10$. (We achieve this by introducing additional 🐮, 🐟, or 🐆 and/or hunting and fishing.) What initial pertubation $x(0)$ would maximize the population of species 1 at time $t=10$? Give the initial perturbation, $x_1(10)$, and plot $x_1(t)$ versus $t$ for $t=0, \ldots, 40$.
"""

# ╔═╡ 3ae53772-e2d8-45e9-b83a-c3a4516d2e37
md"""
Let $B = A^{10}$, so $x(10) = A^{10} x(0) = B x(0)$. We want to maximize $x_1(10)$, which is given by the inner product of the 1st row of $B$ with $x(0)$. The jth term is independent of the rest, and $b_{1,j} x_j (0)$ is maximized when $x_j(0) = \text{sign}(b_{1,j})$, *i.e.*, $x_j(0)$ is either +1 or -1.
"""

# ╔═╡ 547af1d2-f11c-4ea1-b15c-661962b7c49f
B = A^10;

# ╔═╡ 1b449d00-5832-4063-abe7-50fe565aca88
sign.(B[1, :])

# ╔═╡ 9c9ec1d3-1e15-46b1-af6d-efe7b09c93e0
begin 
	x_t = zeros(41, n)
	x_t[1, :] = sign.(B[1, :])
	
	for t = 1:40
		x_t[t+1, :] = A * x_t[t, :]
	end
end

# ╔═╡ 046911e9-602a-45a2-aed9-37721f33f0e7
x_t[11]

# ╔═╡ c20f76d1-5d81-4ecb-84a0-cee9db238de8
plot(0:40, x_t[:,1], label="🐮", xlabel="time t", ylabel="Population deviation x₁ (t)")

# ╔═╡ Cell order:
# ╠═58cbddf4-b723-11eb-03c1-df1b4cec59a4
# ╠═f9bcde53-1070-4c91-b860-dbeef1f7a9c2
# ╟─e0e01781-4c2a-4461-8cb1-d5ff16d9350e
# ╠═40e1dc49-626b-4780-9467-f7da180168de
# ╟─c218188c-1973-4383-b60f-2d22d97fe1e3
# ╠═3ae53772-e2d8-45e9-b83a-c3a4516d2e37
# ╠═547af1d2-f11c-4ea1-b15c-661962b7c49f
# ╠═1b449d00-5832-4063-abe7-50fe565aca88
# ╠═9c9ec1d3-1e15-46b1-af6d-efe7b09c93e0
# ╠═046911e9-602a-45a2-aed9-37721f33f0e7
# ╠═c20f76d1-5d81-4ecb-84a0-cee9db238de8
