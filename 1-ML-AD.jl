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

# ╔═╡ eafb1353-960e-4fc3-b721-6887f0f58059
TableOfContents()   

# ╔═╡ f8ad7eb2-806f-49b7-a6f7-fc220786f10d
md"""
# ROAPA 1 - Learning and Automatic Differentiation
"""

# ╔═╡ fb8d1db6-189a-43ce-934c-fa345034a866
ChooseDisplayMode()

# ╔═╡ 838c61a9-f830-46aa-b788-eadd0b485c8d
md"""
This notebook is a tutorial about deep learning in Julia using the [Flux.jl](https://fluxml.ai/Flux.jl/stable/) library.
An alternative to Flux is the new [Lux.jl](https://lux.csail.mit.edu/) package.

`Flux` is based on the backward automatic differentiation package [Zygote.jl](https://fluxml.ai/Zygote.jl/dev/).

For forward automatic differentiation (to not use with neural networks), [ForwarsdDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/) is the go to package.

For backward autodiff, we recommend Zygote for now, but several alternative exist:
- [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)
- [Enzyme.jl](https://enzymead.github.io/Enzyme.jl/stable/) 
- [Diffractor.jl](https://juliadiff.org/Diffractor.jl/dev/)

Most autodiff packages are based on interfaces provided by [ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/stable/).
"""

# ╔═╡ 9e77b325-660e-4656-b1fc-99b089fd84f3
md"""
## Deep learning pipelines

First, let us load Flux and a plotting library.
"""

# ╔═╡ 0ce32498-937f-4ba9-9841-e7080093d2ab
md"""
We can now create a dense layer:
"""

# ╔═╡ 51ea6004-3856-4ed2-b353-ea7e88f2a4c1
m = Dense(10, 5)

# ╔═╡ 62082401-3e38-4085-8f4a-971661673c98
md"""
We can easily get its parameters
"""

# ╔═╡ 0036ac63-566d-46a2-9cb3-fcc7b8bda6a0
Flux.params(m)

# ╔═╡ 04c8e0be-f6f0-433d-9d1f-9de56ad381d6
md"""which are a weight matrix and a bias vector:"""

# ╔═╡ 08c8ed02-fdd5-408c-8d3c-65f402ddcdbb
m.weight, m.bias

# ╔═╡ ecf1ab63-3e7e-4457-b74f-8bca7d68e4d8
md"""
We can now create a two layer network and take the jacobian with respect to its parameters at point ``x_1``,  as follow:
"""

# ╔═╡ e1f416b9-c240-4a9c-8fba-7e0ac34b1738
begin
	x₁ = rand(Float32, 10)
	m1 = Chain(Dense(10, 5, relu), Dense(5, 2), softmax)
	jac = Flux.jacobian(Flux.params(m1)) do
	    return m1(x₁)
	end
end;

# ╔═╡ 35a3f310-723c-442a-ac3c-a90cc8c2cc8e
md"""To view the jacobian value in `jac`, you can do the following:"""

# ╔═╡ c9902b34-5667-40fa-bacf-f1ba949e0b96
[jac[p] for p in Flux.params(m1)]

# ╔═╡ 2c503b43-c4f7-4680-a84f-a30e7d5bb329
md"""
Let us now train our model. To that purpose we need

- A loss
- An optimizer
- A training set
"""

# ╔═╡ fae3b614-b833-4afa-8c9f-065bc1c83278
m2 = deepcopy(m1)

# ╔═╡ bf02b729-9bc1-4eea-9193-cc3eb4fa35e5
md"""We chose the Adam optimizer, one of the most popular"""

# ╔═╡ 29b5851d-3a09-4613-b93f-4b6cf90baf45
opt = Adam()

# ╔═╡ 3ca170b8-d105-437f-844a-ec25dc6b26e7
md"""We choose the cross entropy loss"""

# ╔═╡ a7279084-9778-49fc-93d3-b034da0401a4
loss(x, y) = Flux.Losses.crossentropy(m2(x), y)

# ╔═╡ b31c26ce-1edf-46de-9061-db1fa8150dab
md"""and build a random training dataset"""

# ╔═╡ 71d7d629-3647-4a95-af42-a777f1587912
data, labels = rand(Float32, 10, 100), fill(0.5f0, 2, 100)

# ╔═╡ 140c9fb6-3dea-46c0-8183-8c618ed50e46
loss(data, labels)

# ╔═╡ 41134cd0-2da7-4b47-bc20-7a750b6a5a51
md"""Gradients can be computed as before:"""

# ╔═╡ 68fee9f2-d8b7-4aee-99aa-ecfa89d7490a
gradient(Flux.params(m2)) do
	loss(data, labels)
end

# ╔═╡ 4bc36c6f-1002-4bd8-bd31-5057106f917b
md"""
We can then write our stochastic gradient descent training loop
"""

# ╔═╡ 3dcfdd79-9bc9-4f4a-8373-386b0b4e89eb
begin
	losses = Float64[]
	for epoch in 1:500
		local l
		grads = gradient(Flux.params(m2)) do
			l = loss(data, labels)
		end
		Flux.update!(opt, Flux.params(m2), grads)
	    push!(losses, l)
	end;
	
end

# ╔═╡ 4297331e-5904-4baf-b281-41116a9c91b7
md"""And plot the training loss evolution"""

# ╔═╡ cdcaf6f7-f829-4f7b-8138-f04d5ace5bff
lineplot(losses)

# ╔═╡ c2c003c3-8696-457a-bd91-fa16c599afb3
md"""
Alternatively, Flux can do all this for you
"""

# ╔═╡ f25acbca-41ca-4350-96f0-022b2409f3c0
begin
	m3 = deepcopy(m1)
	Flux.train!(loss, Flux.params(m3), [(data, labels)], opt)
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
First we import the Automatic Differentiation library, Zygote for backward differentiation, and ForwardDiff for forward mode.
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
Zygote.gradient(f₁, 1.0) 

# ╔═╡ 158b4cba-392d-434f-9a4e-ca7616d526b7
ForwardDiff.derivative(f₁, 1.0)

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
Zygote.gradient(f₂, [1, 1])

# ╔═╡ 5fb975ab-3fef-4237-96e5-017859773140
ForwardDiff.gradient(f₂, [1, 1])

# ╔═╡ aa1b97f3-4f91-489e-b56c-0d13da3eae52
md"""
as well as computing the Jacobian of vector valued functions
"""

# ╔═╡ 8529ee10-317b-43be-8e90-c3f4f12c6918
f₃(x) = [sin(x[1]) + cos(x[2]) + tan(x[3]), exp(x[1])]

# ╔═╡ a599ff9c-fcc7-4ffe-a118-aaf7708dc67b
Zygote.jacobian(f₃, [2, 2, 2])

# ╔═╡ eddf0396-4781-4de1-93a1-6f8f58b10cb5
ForwardDiff.jacobian(f₃, [2, 2, 2])

# ╔═╡ 0cc88686-54d2-4685-982e-017d3b60d612
md"""
### Exercise
"""

# ╔═╡ 85ca0f2f-f57a-4644-b3b6-ef531437174c
question_box(md"Compute the derivative of $x\mapsto \sin\big(1+\cos(1-e^x)\big)$ in $x=7$")

# ╔═╡ 03c94f53-763e-4b86-b5ed-72cc4d5c748e
# TODO derivative

# ╔═╡ 723c27a8-f635-4424-ba77-5bfce66d98c5
question_box(md"Compute the gradient $(x,y) \mapsto \sin(x + \exp(y\cos(x)))$ in $(3,5)$")

# ╔═╡ 7922f2cc-0a3b-4585-872f-944f7d616381
# TODO gradient

# ╔═╡ dd968b2c-dea2-49be-b422-9bc823ce4fe1
question_box(md"Compute the jacobian of $(x,y) \mapsto\left(\begin{array}{c} \tan(x*y) \\ \exp(x-y) \end{array}\right)$ in $(8,3)$")

# ╔═╡ 9e8e9d09-2f29-4bd2-9a54-a9020945763d
# TODO jacobian

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
f₄([3,2,1]), Zygote.gradient(f₄, [3, 2, 1])

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
Zygote.gradient(f₅, [3, 2, 1])

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
	@variable(model, x>= 0)
	@variable(model, 0 <= y <= 5)
	@constraint(model, 2x + 3y >= 5)
	@constraint(model, x + y >= 4)
	@constraint(model, 5x + y >= 7)
	@objective(model, Min, 7x + 9y)
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
	@variable(model, x >= 0)
	@variable(model, 0 <= y <= 5)
	@constraint(model, 2x + 3y >= 5)
	@constraint(model, x + y >= 4)
	@constraint(model, 5x + y >= 7)
	@objective(model, Min, θ[1] * x + θ[2] * y)
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
gradient(f₆, [2.0, 3.0])

# ╔═╡ 3d349d67-f799-4e3f-ae1f-aa92f1c4f24b
md"""
However, it is possible to give an automatic differentation rule to `Julia` for this function. In practice, many automatic differentation libraries are based on `ChainRulesCore.jl`. All we have to do is to provide a rule to differentiate $f$.
"""

# ╔═╡ cfde8cd4-ba51-4724-b9de-e173be20c7f1
import ChainRulesCore

# ╔═╡ fceceb4c-1c36-4172-8b01-c2ca1c3d7312
function f₇(θ)
	model = Model(GLPK.Optimizer)
	@variable(model, x >= 0)
	@variable(model, 0 <= y <= 5)
	@constraint(model, 2x + 3y >= 5)
	@constraint(model, x + y >= 4)
	@constraint(model, 5x + y >= 7)
	@objective(model, Min, θ[1] * x + θ[2] * y)
	optimize!(model)
	return objective_value(model)
end

# ╔═╡ d0c1b95a-4406-458e-83d4-772f224cea17
function ChainRulesCore.rrule(::typeof(f₇), θ)
	model = Model(GLPK.Optimizer)
	@variable(model, x >= 0)
	@variable(model, 0 <= z <= 5)
	@constraint(model, 2z + 3z >= 5)
	@constraint(model, x + z >= 4)
	@constraint(model, 5x + z >= 7)
	@objective(model, Min, θ[1] * x + θ[2] * z)
	optimize!(model)
	y =  objective_value(model)

	function f_pullback(ȳ)
		f̄ = ChainRulesCore.NoTangent()
		θ̄ = [value(x), value(z)] * ȳ
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
Zygote.gradient(f₇, [cos(α), sin(α)])

# ╔═╡ fcae7abe-6d0c-4a77-baee-1f71de4cad87
md"""
Of course, this is compatible with composition.
"""

# ╔═╡ 866fced3-08ee-4101-955c-d48697187d46
Zygote.gradient(θ -> 4 * f₇(2 * θ), [cos(α), sin(α)])

# ╔═╡ 34abfbf8-8b32-4c5b-b458-633be5e0db4d
md"""
### Exercise
"""

# ╔═╡ 3f0e0a89-76a5-4f17-9c92-e226a56a2444
question_box(md"""Implement the following function

```math 
	\begin{array}{rrl}v(\mu) =& \min & x_1 + 3x_2 \\
		&\mathrm{s.t.}
		& x_1+x_2 \geq \theta_1 \\
		&& x_1 + 2x_2 \geq 3 \\
		&& 3x_1 + x_2 \geq \theta_2 \\
		&& x_1,x_2 \geq 0
	\end{array} 
```
""")

# ╔═╡ 93d63480-14b5-47d8-aa1d-fbace34f84ec
function v(μ)
	# TODO
end

# ╔═╡ eef222b4-9019-4d12-9134-8d07cc9fd48c
question_box(md"""Implement the backward rule for $v$.""")

# ╔═╡ 1b342bfc-b035-41a9-8dfa-e4bbe10df218
hint(md"The derivative of the value of an LP with respect to the right hand side of a constraint is equal to the value of the optimal dual of the constraint.")

# ╔═╡ a926561a-e641-47dd-9ffe-43686acc0ca7
function ChainRulesCore.rrule(::typeof(v), μ)
	# TODO
	y = nothing # TODO
	function v_pullback(v̄)
		# TODO
	end
	return y, v_pullback
end

# ╔═╡ ed11f2fa-be74-4f19-8464-beaece595f1b
Zygote.gradient(v, [1., 2.])

# ╔═╡ 22ea9e83-bbef-44b4-afad-b0fc42ca3147
md"""
## Opening the box of Automatic Differentiation
"""

# ╔═╡ 8a91cd46-4c12-40ae-a90c-ac270fd91a2c
md"""
### Backward mode AD

Let us now look a bit more in details at how this works. In practice, `Zygote` performs backward differentiation using rules implemented in `ChainRules`. We are going to compute the derivative of $x \rightarrow 2 (x + a)$ using basic rules. To begin with, let us compute directly the Jacobian with `Zygote`.
"""

# ╔═╡ d342f27e-a1d6-4765-8d25-e97b56bca522
begin
	x₀ = [1, 1, 1] 
	a = [1, 2, 3]
	Zygote.jacobian(x -> 2 * (x + a), x₀)
end

# ╔═╡ d067e573-2565-4743-b0cd-3f51eb3714e7
md"""
In practice, backward AD works as follows.

We start by computing the value of the functions, and at the same time the pullback functions. Suppose that $a : x \mapsto y$, then $\mathtt{pullback}_a(\frac{df}{dy}|_y) \mapsto \frac{df}{dx}|_x$
"""

# ╔═╡ 776665c4-098f-4279-a6f1-3e2520e9bb1c
begin
	y₀, p_pullback = ChainRulesCore.rrule(+, x₀, a)
	z₀, m_pullback = ChainRulesCore.rrule(*, y₀, 2)
end

# ╔═╡ bbe371b3-c0e2-40d3-9c4b-15b1af728d9f
md"""
Backward mode AD, and therefore pullbacks, compute vector-Jacobian products of the form $v^\top J$, hence computing the Jacobian row by row. We are now in a position to compute the first row of the Jacobian. 
"""

# ╔═╡ a379e975-1280-4890-bd0e-c1023997db93
begin
	z̄₁ = [1, 0, 0]
	_, ȳ₁, _ = m_pullback(z̄₁)
	_, x̄₁, _ = p_pullback(ȳ₁)
	x̄₁
end

# ╔═╡ f66ac65c-5212-4ae9-bf91-7e4cb1ed7f08
md"""
Then the second row, etc.
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
The computation now look a little but different: Jacobian is computed column by column
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

# ╔═╡ b9f9ec5d-4b92-444a-997c-57eac0d16b4e
m₁

# ╔═╡ 5630c50b-1545-48f6-a8f7-17804c9de05a
begin
	stats = @timed Zygote.jacobian(m₁, ones(Float32, s_int));
	stats_2 = @timed ForwardDiff.jacobian(m₁, ones(Float32, s_int));
end;

# ╔═╡ f78f42d1-d4c0-454b-b118-fc0b456cc050
md"""
 - BD Zygote took $(stats.time) seconds and $(stats.bytes) bytes
 - FD ForwardDiff took $(stats_2.time) seconds and $(stats_2.bytes) bytes

When the output dimension is much larger than the input dimension, it is more interesting to compute jacobian by column, i.e., using forward AD
"""

# ╔═╡ 4c6ac22e-2265-4c35-980d-cc084cf0cf70
begin
	m₂ = Chain(Dense(1, s_int), relu, Dense(s_int, 1000))
	stats_3 = @timed Zygote.jacobian(m₂, ones(Float32, 1));
	stats_4 = @timed ForwardDiff.jacobian(m₂, ones(Float32, 1));
end;

# ╔═╡ 537e619f-eb9e-4432-a5c0-ade05b16320f
md"""
 - BD Zygote took $(stats_3.time) seconds and $(stats_3.bytes) bytes
 - FD ForwardDiff took $(stats_4.time) seconds and $(stats_4.bytes) bytes

When the input dimension is much larger than the output dimension, it is more interesting to compute jacobian by column, i.e., using backward AD
"""

# ╔═╡ 2a3388cb-0030-4991-b66a-784ce71d35b5
begin
	m₃ = Chain(Dense(1000, s_int), relu, Dense(s_int, 1))
	stats_5 = @timed Zygote.jacobian(m₃, ones(Float32, 1000));
	stats_6 = @timed ForwardDiff.jacobian(m₃, ones(Float32, 1000));
end;

# ╔═╡ 0fb008af-3baa-4ada-bc94-e642c674b81c
md"""
 - BD Zygote took $(stats_5.time) seconds and $(stats_5.bytes) bytes
 - FD ForwardDiff took $(stats_6.time) seconds and $(stats_6.bytes) bytes
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
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
ChainRulesCore = "~1.18.0"
Flux = "~0.14.6"
ForwardDiff = "~0.10.36"
GLPK = "~1.1.3"
JuMP = "~1.16.0"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.54"
UnicodePlots = "~3.6.0"
Zygote = "~0.6.67"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "d74cbd0ffd01cd3a79ceed4099fa48ac14b9eefd"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "006cc7170be3e0fa02ccac6d4164a1eee1fc8c27"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.58.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "c0ae2a86b162fb5d7acc65269b469ff5b8a73594"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "35f0c0f345bff2c6d636f95fdb136323b5a796ef"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.7.0"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "b97c3fc4f3628b8835d83789b09382961a254da4"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.14.6"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxMetalExt = "Metal"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9a68d75d466ccc1218d0552a8e1631151c569545"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "e37c68890d71c2e6555d3689a5d5fc75b35990ef"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.1.3"

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
git-tree-sha1 = "85d7fb51afb3def5dcb85ad31c3707795c8bccc1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "9.1.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "8aa91235360659ca7560db43a7d57541120aa31d"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.11"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "Printf", "SnoopPrecompile", "SparseArrays"]
git-tree-sha1 = "25b2fcda4d455b6f93ac753730d741340ba4a4fe"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.16.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0592b1810613d1c95eeebcd22dc11fba186c2a57"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.26"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "b0737cbbe1c8da6f1139d1c23e35e7cea129c0af"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.13"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "c879e47398a7ab671c782e02b51a4456794a7fa3"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.4.0"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "98eaee04d96d973e79c25d49167668c5c8fb50e2"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.27+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "3504cdb8c2bc05bde4d4b09a81b01df88fcbbba0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.3"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.MarchingCubes]]
deps = ["PrecompileTools", "StaticArrays"]
git-tree-sha1 = "c8e29e2bacb98c9b6f10445227a8b0402f2f173a"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "70ea2892b8bfffecc0387ba1a6a21192814f120c"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.22.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "6985021d02ab8c509c841bb8b2becd3145a7b490"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.3.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "ac86d2944bf7a670ac8bf0f7ec099b5898abcc09"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.8"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "5e4029759e8699ec12ebdf8721e51a659443403c"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.4"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

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
git-tree-sha1 = "34205b1204cc83c43cd9cfe53ffbd3b310f6e8c5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.1"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

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
git-tree-sha1 = "a38e7d70267283888bc83911626961f0b8d5966f"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.9"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "91402087fd5d13b2d97e3ef29bbdf9d7859e678a"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.1"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "5ef59aea6f18c25168842bded46b16662141ab87"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.7.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

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

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "e579d3c991938fecbb225699e8f611fa3fbf2141"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.79"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "LinearAlgebra", "MarchingCubes", "NaNMath", "PrecompileTools", "Printf", "Requires", "SparseArrays", "StaticArrays", "StatsBase"]
git-tree-sha1 = "b96de03092fe4b18ac7e4786bee55578d4b75ae8"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.6.0"

    [deps.UnicodePlots.extensions]
    FreeTypeExt = ["FileIO", "FreeType"]
    ImageInTerminalExt = "ImageInTerminal"
    IntervalSetsExt = "IntervalSets"
    TermExt = "Term"
    UnitfulExt = "Unitful"

    [deps.UnicodePlots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    FreeType = "b38be410-82b0-50bf-ab77-7b57e271db43"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Term = "22787eb5-b846-44ae-b979-8e399b8463ab"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5ded212acd815612df112bb895ef3910c5a03f57"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.67"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "9d749cd449fb448aeca4feee9a2f4186dbb5d184"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.4"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─7573d790-238f-11ed-1e97-bb9bb96aae46
# ╟─eafb1353-960e-4fc3-b721-6887f0f58059
# ╟─f8ad7eb2-806f-49b7-a6f7-fc220786f10d
# ╟─fb8d1db6-189a-43ce-934c-fa345034a866
# ╟─838c61a9-f830-46aa-b788-eadd0b485c8d
# ╟─9e77b325-660e-4656-b1fc-99b089fd84f3
# ╠═577363c7-40e6-4f63-9adb-67d7916d6f32
# ╟─0ce32498-937f-4ba9-9841-e7080093d2ab
# ╠═51ea6004-3856-4ed2-b353-ea7e88f2a4c1
# ╟─62082401-3e38-4085-8f4a-971661673c98
# ╠═0036ac63-566d-46a2-9cb3-fcc7b8bda6a0
# ╟─04c8e0be-f6f0-433d-9d1f-9de56ad381d6
# ╠═08c8ed02-fdd5-408c-8d3c-65f402ddcdbb
# ╠═b9f9ec5d-4b92-444a-997c-57eac0d16b4e
# ╟─ecf1ab63-3e7e-4457-b74f-8bca7d68e4d8
# ╠═e1f416b9-c240-4a9c-8fba-7e0ac34b1738
# ╟─35a3f310-723c-442a-ac3c-a90cc8c2cc8e
# ╠═c9902b34-5667-40fa-bacf-f1ba949e0b96
# ╟─2c503b43-c4f7-4680-a84f-a30e7d5bb329
# ╠═fae3b614-b833-4afa-8c9f-065bc1c83278
# ╟─bf02b729-9bc1-4eea-9193-cc3eb4fa35e5
# ╠═29b5851d-3a09-4613-b93f-4b6cf90baf45
# ╟─3ca170b8-d105-437f-844a-ec25dc6b26e7
# ╠═a7279084-9778-49fc-93d3-b034da0401a4
# ╟─b31c26ce-1edf-46de-9061-db1fa8150dab
# ╠═71d7d629-3647-4a95-af42-a777f1587912
# ╠═140c9fb6-3dea-46c0-8183-8c618ed50e46
# ╟─41134cd0-2da7-4b47-bc20-7a750b6a5a51
# ╠═68fee9f2-d8b7-4aee-99aa-ecfa89d7490a
# ╟─4bc36c6f-1002-4bd8-bd31-5057106f917b
# ╠═3dcfdd79-9bc9-4f4a-8373-386b0b4e89eb
# ╟─4297331e-5904-4baf-b281-41116a9c91b7
# ╠═cdcaf6f7-f829-4f7b-8138-f04d5ace5bff
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
# ╠═158b4cba-392d-434f-9a4e-ca7616d526b7
# ╟─f5db9a84-2594-42cf-916c-32a3d8c4a16f
# ╠═aefb1fcf-20a0-4cca-a86b-9b6c38aa020c
# ╟─1a3cd09a-7a57-4134-b495-e7888914b6e0
# ╠═9320d96e-413d-45d2-b419-ed5b52360043
# ╠═5fb975ab-3fef-4237-96e5-017859773140
# ╟─aa1b97f3-4f91-489e-b56c-0d13da3eae52
# ╠═8529ee10-317b-43be-8e90-c3f4f12c6918
# ╠═a599ff9c-fcc7-4ffe-a118-aaf7708dc67b
# ╠═eddf0396-4781-4de1-93a1-6f8f58b10cb5
# ╟─0cc88686-54d2-4685-982e-017d3b60d612
# ╟─85ca0f2f-f57a-4644-b3b6-ef531437174c
# ╠═03c94f53-763e-4b86-b5ed-72cc4d5c748e
# ╟─723c27a8-f635-4424-ba77-5bfce66d98c5
# ╠═7922f2cc-0a3b-4585-872f-944f7d616381
# ╟─dd968b2c-dea2-49be-b422-9bc823ce4fe1
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
# ╟─47138cd1-121b-47f2-8252-6621148e95d5
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
# ╠═34abfbf8-8b32-4c5b-b458-633be5e0db4d
# ╟─3f0e0a89-76a5-4f17-9c92-e226a56a2444
# ╠═93d63480-14b5-47d8-aa1d-fbace34f84ec
# ╟─eef222b4-9019-4d12-9134-8d07cc9fd48c
# ╟─1b342bfc-b035-41a9-8dfa-e4bbe10df218
# ╠═a926561a-e641-47dd-9ffe-43686acc0ca7
# ╠═ed11f2fa-be74-4f19-8464-beaece595f1b
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
