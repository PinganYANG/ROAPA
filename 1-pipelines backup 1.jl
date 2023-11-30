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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
GLPK = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
ChainRules = "~1.44.6"
ChainRulesCore = "~1.15.5"
Flux = "~0.13.6"
ForwardDiff = "~0.10.32"
GLPK = "~1.1.0"
JuMP = "~1.3.0"
PlutoTeachingTools = "~0.2.3"
PlutoUI = "~0.7.43"
UnicodePlots = "~3.1.2"
Zygote = "~0.6.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.1"
manifest_format = "2.0"
project_hash = "358cecef3deda95fc2a0708fb400ce7c26b91062"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "d6173480145eb632d6571c148d94b9d3d773820e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.23"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5bb0f8292405a516880a3809954cb832ae7a31c5"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.20"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "49549e2c28ffb9cc77b3689dc10e46e6271e9452"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.12.0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "a5fd229d3569a6600ae47abe8cd48cbeb972e173"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.44.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "dc4405cee4b2fe9e1108caec2d760b7ea758eca2"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.5"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "1833bda4a027f4b2a1c984baddcf755d77266818"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.1.0"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "1106fa7e1256b402a86a8e7b15c00c85036fef49"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.11.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "992a23afdb109d0d2f8802a30cf5ae4b1fe7ea68"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "94f5101b96d2d968ace56f7f2db19d0a5f592e28"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.15.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "87519eb762f85534445f5cda35be12e32759ee14"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.4"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ArrayInterface", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "OneHotArrays", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "76ca02c7c0cb7b8337f7d2d0eadb46ed03c1e843"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.6"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "38a92e40157100e796690421e34a11c107205c86"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "a2657dd0f3e8a61dbe70fc7c122038bd33790af5"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.3.0"

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "e357b935632e89a02cf7f8f13b4f3f59cef479c8"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.1.0"

[[deps.GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "fe68622f32828aa92275895fdb324a85894a5b1b"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.2.1+2"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "45d7deaf05cbb44116ba785d147c518ab46352d7"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.5.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "ebb892e1df16040a845e1d11087e4fbfe10323a8"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.4"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "12a584db96f1d460421d5fb8860822971cdb8455"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays"]
git-tree-sha1 = "906e2325c22ba8aaed432677d0a8d5cf24c9ea9e"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.3.0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0f960b1404abb0b244c1ece579a0ec78d056a5d1"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.15"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "Random", "ShowCases", "Statistics", "StatsBase"]
git-tree-sha1 = "c92a10a2492dffac0e152a19d5ffd99a5030349a"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.1"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MarchingCubes]]
deps = ["SnoopPrecompile", "StaticArrays"]
git-tree-sha1 = "ffc66942498a5f0d02b9e7b1b1af0f5873142cdc"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "2284cb18c8670fd5c57ad010ce9bd4e2901692d2"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.8.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4e675d6e9ec02061800d6cfb695812becbd03cdf"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.4"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "415108fd88d6f55cedf7ee940c7d4b01fad85421"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.9"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "4429261364c5ea5b7308aecaa10e803ace101631"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "GPUArrays", "LinearAlgebra", "MLUtils", "NNlib"]
git-tree-sha1 = "2f6efe2f76d57a0ee67cb6eff49b4d02fccbd175"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.1.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1ef34738708e3f31994b52693286dabcb3d29f6b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.9"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "0e8bcc235ec8367a8e9648d48325ff00e4b0a545"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.5"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "d8be3432505c2febcea02f44e5f4396fae017503"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "2777a5c2c91b3145f5aa75b61bb4c2eb38797136"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.43"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "7a1a306b72cfa60634f03a911405f4e64d1b718b"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "dad726963ecea2d8a81e26286f625aee09a91b7c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.4.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "de4f0a4f049a4c87e4948c04acff37baf1be01a6"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.7.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2d4e51cfad63d2d34acde558027acbc66700349b"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.3"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "7149a60b01bf58787a1b83dad93f90d4b9afbe5d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.8.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "9dfcb767e17b0849d6aaf85997c98a5aea292513"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.21"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeTypeAbstraction", "LazyModules", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "Requires", "SnoopPrecompile", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "f2ac653d1b971c27f59c1ba88532ca3c259031e2"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.1.2"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d57a4ed70b6f9ff1da6719f5f2713706d57e0d66"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "66cc604b9a27a660e25a54e408b4371123a186a6"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.49"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
