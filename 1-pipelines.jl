### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 7573d790-238f-11ed-1e97-bb9bb96aae46
using PlutoUI, PlutoTeachingTools

# ╔═╡ 577363c7-40e6-4f63-9adb-67d7916d6f32
using Flux, UnicodePlots

# ╔═╡ f530731d-cb6c-410e-b066-381e9762b8d4
using GLPK, JuMP

# ╔═╡ f8ad7eb2-806f-49b7-a6f7-fc220786f10d
md"""
# ROAPA 1 - pipelines
"""

# ╔═╡ fb8d1db6-189a-43ce-934c-fa345034a866
ChooseDisplayMode()

# ╔═╡ eafb1353-960e-4fc3-b721-6887f0f58059
TableOfContents()   

# ╔═╡ 9e77b325-660e-4656-b1fc-99b089fd84f3
md"""
## Deep learning pipelines

First, let us load the library
"""

# ╔═╡ 0ce32498-937f-4ba9-9841-e7080093d2ab
md"""
We can now create a dense layer
"""

# ╔═╡ 51ea6004-3856-4ed2-b353-ea7e88f2a4c1
m = Dense(10, 5)

# ╔═╡ 62082401-3e38-4085-8f4a-971661673c98
md"""
We can easily get its parameters
"""

# ╔═╡ 0036ac63-566d-46a2-9cb3-fcc7b8bda6a0
Flux.params(m)

# ╔═╡ ecf1ab63-3e7e-4457-b74f-8bca7d68e4d8
md"""
We can now create a two layer network and take the gradients with respect to its parameters
"""

# ╔═╡ e1f416b9-c240-4a9c-8fba-7e0ac34b1738
begin
	x_1 = rand(Float32, 10)
	m_1 = Chain(Dense(10, 5, relu), Dense(5, 2), softmax)
	l(x) = Flux.Losses.crossentropy(m_1(x), [0.5, 0.5])
	grads = gradient(Flux.params(m_1)) do
	    l(x_1)
	end
end

# ╔═╡ 2c503b43-c4f7-4680-a84f-a30e7d5bb329
md"""
Let us now train our model. To that purpose we need

- A loss
- An optimizer
- A training set
"""

# ╔═╡ 140c9fb6-3dea-46c0-8183-8c618ed50e46
begin
	opt = ADAM()
	m_2 = deepcopy(m_1)
	loss(x, y) = Flux.Losses.crossentropy(m_2(x), y)
	data, labels = rand(10, 100), fill(0.5, 2, 100)
	loss(data,labels)
end

# ╔═╡ 68fee9f2-d8b7-4aee-99aa-ecfa89d7490a
gradient(Flux.params(m_2)) do
	loss(data, labels)
end

# ╔═╡ 4bc36c6f-1002-4bd8-bd31-5057106f917b
md"""
We can then write our stochastic gradient descent
"""

# ╔═╡ 3dcfdd79-9bc9-4f4a-8373-386b0b4e89eb
begin
	losses = Float64[]
	for epoch in 1:200
		grads = gradient(Flux.params(m_2)) do
			l = loss(data, labels)
		end
		Flux.update!(opt, Flux.params(m_2), grads)
	    push!(losses, l)
	end;
	lineplot(losses)
end

# ╔═╡ c2c003c3-8696-457a-bd91-fa16c599afb3
md"""
Flux can do all this for you
"""

# ╔═╡ f25acbca-41ca-4350-96f0-022b2409f3c0
begin
	m_3 = deepcopy(m_1)
	Flux.train!(loss, Flux.params(m_2), [(data,labels)], opt)
end

# ╔═╡ 1f56cb44-142c-42cc-8fe7-5fed606b4361
md"""
## Automatic differentiation
"""

# ╔═╡ 432e6c8b-2479-4732-8cd3-1f0311ad7e21
md"""
### Automatic differentiation of simple functions
"""

# ╔═╡ 2754f850-0039-4238-9660-5a69b2761343
md"""
First we import the Automatic Differentiation library. We can then take the derivative of a function without giving manually the code to compute it.
Here, we are going to use `Zygote`, which is based on backward automatic differentiation. There are alternatives, among which `ForwardDiff` that performs forward automatic differentation.
"""

# ╔═╡ 1cb8f863-b222-460f-99b7-c90b7084df81
import Zygote, ForwardDiff

# ╔═╡ 3bfe7db1-0e25-4c0b-ad14-eaf6ff115d5e
md"""
Let us now define a function $f_1$
"""

# ╔═╡ 2eaae6ba-5dea-44e6-af11-b88c67767f33
f₁(x) = x^2

# ╔═╡ a78339e1-a55c-452f-8896-ee1960cc6edd
md"""
and compute its derivative using both `Zygote` and `ForwardDiff`
"""

# ╔═╡ 14101b99-a388-41f3-b3bc-3143d3ec93dd
Zygote.gradient(f₁,1), ForwardDiff.derivative(f₁,1)

# ╔═╡ f5db9a84-2594-42cf-916c-32a3d8c4a16f
md"""
Of course, we can deal with functions of several variables
"""

# ╔═╡ aefb1fcf-20a0-4cca-a86b-9b6c38aa020c
f₂(x) = sin(x[1]) + cos(x[2])

# ╔═╡ 1a3cd09a-7a57-4134-b495-e7888914b6e0
md"""
and compute their gradient
"""

# ╔═╡ 9320d96e-413d-45d2-b419-ed5b52360043
Zygote.gradient(f₂,[1,1]), ForwardDiff.gradient(f₂,[1,1,])

# ╔═╡ aa1b97f3-4f91-489e-b56c-0d13da3eae52
md"""
as well as computing the Jacobian of vector valued functions
"""

# ╔═╡ 8529ee10-317b-43be-8e90-c3f4f12c6918
f₃(x) = [sin(x[1]) + cos(x[2]) + tan(x[3]), exp(x[1])]

# ╔═╡ a599ff9c-fcc7-4ffe-a118-aaf7708dc67b
Zygote.jacobian(f₃,[2,2,2]), ForwardDiff.jacobian(f₃,[2,2,2])

# ╔═╡ 0cc88686-54d2-4685-982e-017d3b60d612
md"""
### Exercise

Compute 

- the derivative of $x\mapsto \sin\big(1+\cos(1-e^x)\big)$ in $x=7$
- the gradient $(x,y) \mapsto \sin(x + \exp(y\cos(x)))$ in $(3,5)$
- the jacobian of $(x,y) \mapsto\left(\begin{array}{c} \tan(x*y) \\ \exp(x-y) \end{array}\right)$ in $(8,3)$
"""

# ╔═╡ 03c94f53-763e-4b86-b5ed-72cc4d5c748e
# TODO derivative
begin
	f_3(x) = sin(1 + cos(1 - exp(x)))
	Zygote.gradient(f_3, 7)
end

# ╔═╡ 7922f2cc-0a3b-4585-872f-944f7d616381
# TODO gradient
begin
	f_4(x) = sin(x[1] + exp(x[2] * cos(x[2])))
	Zygote.gradient(f_4, [3, 5])
end

# ╔═╡ 9e8e9d09-2f29-4bd2-9a54-a9020945763d
# TODO jacobian
begin
	f_5(x) = [tan(x[1] * x[2]), tan(x[1] - x[2])]
	Zygote.gradient(f_4, [8, 3])
end

# ╔═╡ 65479539-9db6-40e1-bf00-9c74b80e091b
md"""
### Automatic differentiation of non-elementary functions
"""

# ╔═╡ b52302d5-89a7-4dad-9c7f-f053d466f648
md"""
Automatic differentiation continues to work with non-trivial functions that combine several steps.
"""

# ╔═╡ 18851432-62d0-44bd-b7b5-5c7b982300b7
function f₄(x)
	y = x - ones(length(x))
	return y' * x
end

# ╔═╡ e99351fe-de58-43d1-8d2b-f2c1b1da9021
f₄([3,2,1]), Zygote.gradient(f₄,[3,2,1])

# ╔═╡ 8cb12f7d-1db4-417e-aa42-bb2d9012e21e
md"""
However, mutating arrays is not supported. This is an implementation choice and not a generic problem linked to automatic differentiation.
"""

# ╔═╡ c2bdf1c1-097c-49c1-b64b-e1d7bd5e6b01
function f₅(x)
	y = x
	y[1] = 4
	return y' * x
end

# ╔═╡ 27860066-29c8-476d-ae36-6f92d680373d
Zygote.gradient(f₅,[3,2,1])

# ╔═╡ 0b953b2c-8f28-474e-8e2a-4ad116c020c5
md"""
### Automatic differentiation of oracles
"""

# ╔═╡ 8f6e4f0a-6a18-471d-ae80-5e68a5516a1c
md"""Let us now call an external LP solver"""

# ╔═╡ 47138cd1-121b-47f2-8252-6621148e95d5
md"""With JuMP, we can formulate a LP, while GLPK is an external solver than can find its optimal solution."""

# ╔═╡ 7a291c45-a244-421b-a842-608810a77e72
begin
	model = Model(GLPK.Optimizer)
	@variable(model,x>=0)
	@variable(model,0<=y<=5)
	@constraint(model,2*x + 3* y >= 5)
	@constraint(model, x + y >= 4)
	@constraint(model, 5x + y >= 7)
	@objective(model, Min, 7*x + 9*y)
	print(model)
end

# ╔═╡ 5deebdd3-d5e3-472c-b88b-b60aab0544af
md"""
GLPK is an external solver than can find its optimal solution.
"""

# ╔═╡ 752b4e90-c1a3-4b29-ab44-e7e58aad9bdf
optimize!(model)

# ╔═╡ e08a3d8d-aa88-4a49-8b5e-eec37591f8c3
md"""
We can then retrieve its objective value and its optimal solution.
"""

# ╔═╡ c19d3b69-00fd-4fc1-9aaf-802664d8d3fc
objective_value(model)

# ╔═╡ dce16f37-34f8-4284-a078-0f66d947ad5e
value(x)

# ╔═╡ 38dbfe9b-8a8e-4424-b1e5-ddb2f5cd5901
value(y)

# ╔═╡ 98783457-4084-4410-be86-ad9fccfae043
md"""
Let us now define a function with solves an LP as a subroutine.
"""

# ╔═╡ 95bce91d-e9d2-49f2-bf71-5192494c26ba
function f₆(θ)
	model = Model(GLPK.Optimizer)
	@variable(model,x>=0)
	@variable(model,0<=y<=5)
	@constraint(model,2*x + 3* y >= 5)
	@constraint(model, x + y >= 4)
	@constraint(model, 5x + y >= 7)
	@objective(model, Min, θ[1]*x + θ[2]*y)
	optimize!(model)
	return objective_value(model)
end

# ╔═╡ 85460997-0414-42b6-8e2b-eea3547a1e34
f₆([2.0, 3.0])

# ╔═╡ b9bd0ddf-136a-4df2-a865-29c5f758f7cc
md"""
As for mutating array, taking the gradient of `f` does not work out of the box because automatic differentiation does not work using an external solver. Indeed, since the code of the black-box is by definition not definef in Julia, there is no way to derive it.
"""

# ╔═╡ 0b184a43-bedd-45c5-85eb-6cfccf8e8aa9
gradient(f₆,[2.0, 3.0])

# ╔═╡ 3d349d67-f799-4e3f-ae1f-aa92f1c4f24b
md"""
However, it is possible to teaching given an automatic differentation rule to `Julia` for this function. In practice, many automatic differentation libraries are based on `ChainRules.jl`. All we have to do is to provide a rule to differentiate $f$.
"""

# ╔═╡ cfde8cd4-ba51-4724-b9de-e173be20c7f1
import ChainRulesCore

# ╔═╡ fceceb4c-1c36-4172-8b01-c2ca1c3d7312
function f₇(θ)
	model = Model(GLPK.Optimizer)
	@variable(model,x>=0)
	@variable(model,0<=y<=5)
	@constraint(model,2*x + 3* y >= 5)
	@constraint(model, x + y >= 4)
	@constraint(model, 5x + y >= 7)
	@objective(model, Min, θ[1]*x + θ[2]*y)
	optimize!(model)
	return objective_value(model)
end

# ╔═╡ d0c1b95a-4406-458e-83d4-772f224cea17
function ChainRulesCore.rrule(::typeof(f₇), θ)

	model = Model(GLPK.Optimizer)
	@variable(model,x>=0)
	@variable(model,0<=z<=5)
	@constraint(model,2*z + 3* z >= 5)
	@constraint(model, x + z >= 4)
	@constraint(model, 5x + z >= 7)
	@objective(model, Min, θ[1]*x + θ[2]*z)
	optimize!(model)
	y =  objective_value(model)

	# using y = f(θ), 
	function f_pullback(ȳ)
		f̄ = ChainRulesCore.NoTangent()
		θ̄ = [value(x),value(z)] * ȳ
		return f̄, θ̄
	end
	
	return y, f_pullback
end

# ╔═╡ a8c402c9-b964-47ad-bffa-b6ca8ac9b73c
@bind α Slider(0:π/100:π/2)

# ╔═╡ a5d3158d-2018-43f8-84eb-de0a9adeeb32
md"""
We can now compute gradients of $f$. 
"""

# ╔═╡ f1529c5c-77b3-42d2-a625-5a9ab950e553
α

# ╔═╡ ca8231ee-9b40-4d2a-9b63-9df7d9928166
Zygote.gradient(f₇,[cos(α),sin(α)])

# ╔═╡ fcae7abe-6d0c-4a77-baee-1f71de4cad87
md"""
Of course, this is compatible with composition.
"""

# ╔═╡ 866fced3-08ee-4101-955c-d48697187d46
Zygote.gradient(θ -> 4 * f₇(2*θ),[cos(α),sin(α)])

# ╔═╡ 34abfbf8-8b32-4c5b-b458-633be5e0db4d
md"""
### Exercise

Implement the following function

```math 
	\begin{array}{rrl}v(\mu) =& \min & x_1 + 3x_2 \\
		&\mathrm{s.t.}
		& x_1+x_2 >= \theta_1 \\
		&& x_1 + 2x_2 \geq 3 \\
		&& 3x_1 + x_2 \geq \theta_2 \\
		&& x_1,x_2 \geq 0
	\end{array} 
```

Implement the backward rule for $f$.
"""

# ╔═╡ 1b342bfc-b035-41a9-8dfa-e4bbe10df218
hint(md"The derivative of the value of an LP with respect to the right hand side of a constraint is equal to the value of the optimal dual of the constraint.")

# ╔═╡ 93d63480-14b5-47d8-aa1d-fbace34f84ec
function v(μ)
	# TODO
end

# ╔═╡ a926561a-e641-47dd-9ffe-43686acc0ca7
function ChainRulesCore.rrule(::typeof(v), μ)
	# TODO
	y= 0 # TODO
	function v_pullback(v̄)
		# TODO
	end
	return y, v_pullback
end

# ╔═╡ 22ea9e83-bbef-44b4-afad-b0fc42ca3147
md"""
## Opening the box of Automatic Differentiation
"""

# ╔═╡ 8a91cd46-4c12-40ae-a90c-ac270fd91a2c
md"""
### Backward mode AD

Let us now look a bit more in details at how this works. In practice, `Zygote` performs backward differentiation using rules implemented in `ChainRules`. We are going to compute the derivative of $x \rightarrow 2 * (x + a)$ using basic rules. To begin with, let us compute directly the Jacobian with `Zygote`.
"""

# ╔═╡ d342f27e-a1d6-4765-8d25-e97b56bca522
begin
	x₀= [1,1,1] 
	a = [1,2,3]
	Zygote.jacobian(x -> 2*(x + a),x₀)
end

# ╔═╡ 776665c4-098f-4279-a6f1-3e2520e9bb1c
begin
	using ChainRules
	y₀, p_pullback = ChainRulesCore.rrule(+,x₀,a)
	z₀, m_pullback = ChainRulesCore.rrule(*,y₀,2)
end

# ╔═╡ d067e573-2565-4743-b0cd-3f51eb3714e7
md"""
In practice, backward AD works as follows.

We start by computing the value of the functions, and at the same time the pullback functions. Suppose that $a : x \mapsto y$, then $\mathtt{pullback}_a(\frac{df}{dy}|_y) \mapsto \frac{df}{dx}|_x$
"""

# ╔═╡ bbe371b3-c0e2-40d3-9c4b-15b1af728d9f
md"""
Backward mode AD, and therefore pullbacks, compute Jacobian vector products of the form $Jv$, hence computing the Jacobian column by column. We are now in a position to compute the first column of the Jacobian. 
"""

# ╔═╡ a379e975-1280-4890-bd0e-c1023997db93
begin
	z̄₁ = [1,0,0]
	_, ȳ₁, _ = m_pullback(z̄₁)
	_, x̄₁, _ = p_pullback(ȳ₁)
	x̄₁
end

# ╔═╡ f66ac65c-5212-4ae9-bf91-7e4cb1ed7f08
md"""
Then the second column, etc.
"""

# ╔═╡ 28bda40e-439f-4ac7-993a-377abc56503e
begin
	z̄₂ = [0,1,0]
	_, ȳ₂, _ = m_pullback(z̄₂)
	_, x̄₂, _ = p_pullback(ȳ₂)
	x̄₂
end

# ╔═╡ 97c9fcf6-550b-4466-a22e-b4085d5fc6f8
md"""
### Forward mode AD

Let us now see what happens in forward mode AD. For instance, we can use `ForwardDiff`
"""

# ╔═╡ 7793762c-73e5-4a46-98ee-86936ac9380c
ForwardDiff.derivative(x -> cos(sin(x)),1)

# ╔═╡ 1988ec7c-950f-4346-b790-3ced3563c283
md"""
The computation now look a little but different: Jacobian is compute row by row
"""

# ╔═╡ eb3e9686-6d8a-4eb1-ba2d-2528d6c931e5
begin
	sinx , Δsinx = ChainRulesCore.frule((ChainRulesCore.NoTangent(),1),sin,1)
	cossinx, Δcosinx = ChainRulesCore.frule((ChainRulesCore.NoTangent(),Δsinx),cos,sinx)
	Δcosinx
end

# ╔═╡ 57f3b280-7159-4088-a119-4d330ad9785f
md"""
Forward mode AD computes vector Jacobian product of the for $u^T J$, hence computing the Jacobian row by row.
"""

# ╔═╡ 7b34ab71-0a9a-4a9a-bff8-24bfa214223e
md"""
### Forward vs backward AD

When the dimension of the input and the output of the pipeline are roughly similar, Forward AD is slightly faster than backward AD
"""

# ╔═╡ 46d2c91a-fbae-40ed-9bb8-1bb46785b5e5
begin
	s_int = 200
	m₁ = Chain(Dense(s_int,s_int),relu,Dense(s_int,s_int))
end

# ╔═╡ 5630c50b-1545-48f6-a8f7-17804c9de05a
begin
	stats = @timed Zygote.jacobian(m₁,ones(s_int));
	stats_2 = @timed ForwardDiff.jacobian(m₁,ones(s_int));
end

# ╔═╡ f78f42d1-d4c0-454b-b118-fc0b456cc050
md"""
 - BD Zygote took $(stats.time) seconds and $(stats.bytes) bytes
 - FD ForwardDiff took $(stats_2.time) seconds and $(stats_2.bytes) bytes

When the output dimension is much larger than the input dimension, it is more interesting to compute jacobian by raw, i.e., using forward AD
"""

# ╔═╡ 4c6ac22e-2265-4c35-980d-cc084cf0cf70
begin
	m₂ = Chain(Dense(1,s_int),relu,Dense(s_int,1000))
	stats_3 = @timed Zygote.jacobian(m₂,ones(1));
	stats_4 = @timed ForwardDiff.jacobian(m₂,ones(1));
end

# ╔═╡ 537e619f-eb9e-4432-a5c0-ade05b16320f
md"""
 - BD Zygote took $(stats_3.time) seconds and $(stats_3.bytes) bytes
 - FD ForwardDiff took $(stats_4.time) seconds and $(stats_4.bytes) bytes

When the input dimension is much larger than the output dimension, it is more interesting to compute jacobian by column, i.e., using backward AD
"""

# ╔═╡ 2a3388cb-0030-4991-b66a-784ce71d35b5
begin
	m₃ = Chain(Dense(1000,s_int),relu,Dense(s_int,1))
	stats_5 = @timed Zygote.jacobian(m₃,ones(1000));
	stats_6 = @timed ForwardDiff.jacobian(m₃,ones(1000));
end

# ╔═╡ 0fb008af-3baa-4ada-bc94-e642c674b81c
md"""
 - BD Zygote took $(stats_5.time) seconds and $(stats_5.bytes) bytes
 - FD ForwardDiff took $(stats_6.time) seconds and $(stats_6.bytes) bytes
"""

# ╔═╡ Cell order:
# ╟─f8ad7eb2-806f-49b7-a6f7-fc220786f10d
# ╠═7573d790-238f-11ed-1e97-bb9bb96aae46
# ╠═fb8d1db6-189a-43ce-934c-fa345034a866
# ╠═eafb1353-960e-4fc3-b721-6887f0f58059
# ╟─9e77b325-660e-4656-b1fc-99b089fd84f3
# ╠═577363c7-40e6-4f63-9adb-67d7916d6f32
# ╟─0ce32498-937f-4ba9-9841-e7080093d2ab
# ╠═51ea6004-3856-4ed2-b353-ea7e88f2a4c1
# ╟─62082401-3e38-4085-8f4a-971661673c98
# ╠═0036ac63-566d-46a2-9cb3-fcc7b8bda6a0
# ╟─ecf1ab63-3e7e-4457-b74f-8bca7d68e4d8
# ╠═e1f416b9-c240-4a9c-8fba-7e0ac34b1738
# ╠═2c503b43-c4f7-4680-a84f-a30e7d5bb329
# ╠═140c9fb6-3dea-46c0-8183-8c618ed50e46
# ╠═68fee9f2-d8b7-4aee-99aa-ecfa89d7490a
# ╟─4bc36c6f-1002-4bd8-bd31-5057106f917b
# ╠═3dcfdd79-9bc9-4f4a-8373-386b0b4e89eb
# ╟─c2c003c3-8696-457a-bd91-fa16c599afb3
# ╠═f25acbca-41ca-4350-96f0-022b2409f3c0
# ╟─1f56cb44-142c-42cc-8fe7-5fed606b4361
# ╟─432e6c8b-2479-4732-8cd3-1f0311ad7e21
# ╟─2754f850-0039-4238-9660-5a69b2761343
# ╠═1cb8f863-b222-460f-99b7-c90b7084df81
# ╟─3bfe7db1-0e25-4c0b-ad14-eaf6ff115d5e
# ╠═2eaae6ba-5dea-44e6-af11-b88c67767f33
# ╟─a78339e1-a55c-452f-8896-ee1960cc6edd
# ╠═14101b99-a388-41f3-b3bc-3143d3ec93dd
# ╟─f5db9a84-2594-42cf-916c-32a3d8c4a16f
# ╠═aefb1fcf-20a0-4cca-a86b-9b6c38aa020c
# ╟─1a3cd09a-7a57-4134-b495-e7888914b6e0
# ╠═9320d96e-413d-45d2-b419-ed5b52360043
# ╟─aa1b97f3-4f91-489e-b56c-0d13da3eae52
# ╠═8529ee10-317b-43be-8e90-c3f4f12c6918
# ╠═a599ff9c-fcc7-4ffe-a118-aaf7708dc67b
# ╟─0cc88686-54d2-4685-982e-017d3b60d612
# ╠═03c94f53-763e-4b86-b5ed-72cc4d5c748e
# ╠═7922f2cc-0a3b-4585-872f-944f7d616381
# ╠═9e8e9d09-2f29-4bd2-9a54-a9020945763d
# ╟─65479539-9db6-40e1-bf00-9c74b80e091b
# ╟─b52302d5-89a7-4dad-9c7f-f053d466f648
# ╠═18851432-62d0-44bd-b7b5-5c7b982300b7
# ╠═e99351fe-de58-43d1-8d2b-f2c1b1da9021
# ╟─8cb12f7d-1db4-417e-aa42-bb2d9012e21e
# ╠═c2bdf1c1-097c-49c1-b64b-e1d7bd5e6b01
# ╠═27860066-29c8-476d-ae36-6f92d680373d
# ╟─0b953b2c-8f28-474e-8e2a-4ad116c020c5
# ╟─8f6e4f0a-6a18-471d-ae80-5e68a5516a1c
# ╠═f530731d-cb6c-410e-b066-381e9762b8d4
# ╠═47138cd1-121b-47f2-8252-6621148e95d5
# ╠═7a291c45-a244-421b-a842-608810a77e72
# ╟─5deebdd3-d5e3-472c-b88b-b60aab0544af
# ╠═752b4e90-c1a3-4b29-ab44-e7e58aad9bdf
# ╟─e08a3d8d-aa88-4a49-8b5e-eec37591f8c3
# ╠═c19d3b69-00fd-4fc1-9aaf-802664d8d3fc
# ╠═dce16f37-34f8-4284-a078-0f66d947ad5e
# ╠═38dbfe9b-8a8e-4424-b1e5-ddb2f5cd5901
# ╟─98783457-4084-4410-be86-ad9fccfae043
# ╠═95bce91d-e9d2-49f2-bf71-5192494c26ba
# ╠═85460997-0414-42b6-8e2b-eea3547a1e34
# ╟─b9bd0ddf-136a-4df2-a865-29c5f758f7cc
# ╠═0b184a43-bedd-45c5-85eb-6cfccf8e8aa9
# ╟─3d349d67-f799-4e3f-ae1f-aa92f1c4f24b
# ╠═cfde8cd4-ba51-4724-b9de-e173be20c7f1
# ╠═fceceb4c-1c36-4172-8b01-c2ca1c3d7312
# ╠═d0c1b95a-4406-458e-83d4-772f224cea17
# ╠═a8c402c9-b964-47ad-bffa-b6ca8ac9b73c
# ╟─a5d3158d-2018-43f8-84eb-de0a9adeeb32
# ╠═f1529c5c-77b3-42d2-a625-5a9ab950e553
# ╠═ca8231ee-9b40-4d2a-9b63-9df7d9928166
# ╟─fcae7abe-6d0c-4a77-baee-1f71de4cad87
# ╠═866fced3-08ee-4101-955c-d48697187d46
# ╟─34abfbf8-8b32-4c5b-b458-633be5e0db4d
# ╟─1b342bfc-b035-41a9-8dfa-e4bbe10df218
# ╠═93d63480-14b5-47d8-aa1d-fbace34f84ec
# ╠═a926561a-e641-47dd-9ffe-43686acc0ca7
# ╟─22ea9e83-bbef-44b4-afad-b0fc42ca3147
# ╟─8a91cd46-4c12-40ae-a90c-ac270fd91a2c
# ╠═d342f27e-a1d6-4765-8d25-e97b56bca522
# ╟─d067e573-2565-4743-b0cd-3f51eb3714e7
# ╠═776665c4-098f-4279-a6f1-3e2520e9bb1c
# ╟─bbe371b3-c0e2-40d3-9c4b-15b1af728d9f
# ╠═a379e975-1280-4890-bd0e-c1023997db93
# ╟─f66ac65c-5212-4ae9-bf91-7e4cb1ed7f08
# ╠═28bda40e-439f-4ac7-993a-377abc56503e
# ╟─97c9fcf6-550b-4466-a22e-b4085d5fc6f8
# ╠═7793762c-73e5-4a46-98ee-86936ac9380c
# ╟─1988ec7c-950f-4346-b790-3ced3563c283
# ╠═eb3e9686-6d8a-4eb1-ba2d-2528d6c931e5
# ╟─57f3b280-7159-4088-a119-4d330ad9785f
# ╟─7b34ab71-0a9a-4a9a-bff8-24bfa214223e
# ╠═46d2c91a-fbae-40ed-9bb8-1bb46785b5e5
# ╠═5630c50b-1545-48f6-a8f7-17804c9de05a
# ╟─f78f42d1-d4c0-454b-b118-fc0b456cc050
# ╠═4c6ac22e-2265-4c35-980d-cc084cf0cf70
# ╟─537e619f-eb9e-4432-a5c0-ade05b16320f
# ╠═2a3388cb-0030-4991-b66a-784ce71d35b5
# ╟─0fb008af-3baa-4ada-bc94-e642c674b81c
