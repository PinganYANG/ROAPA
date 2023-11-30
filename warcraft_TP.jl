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

# ╔═╡ 6160785a-d455-40b4-ab74-e61c46e31537
# ╠═╡ show_logs = false
begin
	import Pkg
    Pkg.activate(mktempdir())
    Pkg.add(Pkg.PackageSpec(name="Colors"))
	Pkg.add(Pkg.PackageSpec(name="CSV"))
	Pkg.add(Pkg.PackageSpec(name="Flux"))
	Pkg.add(Pkg.PackageSpec(name="GZip"))
	Pkg.add(Pkg.PackageSpec(name="Graphs"))
	Pkg.add(Pkg.PackageSpec(url="https://github.com/gdalle/GridGraphs.jl.git"))
	Pkg.add(Pkg.PackageSpec(name="Images"))
	Pkg.add(Pkg.PackageSpec(name="InferOpt"))
	Pkg.add(Pkg.PackageSpec(name="JSON"))
	Pkg.add(Pkg.PackageSpec(name="LinearAlgebra"))
	Pkg.add(Pkg.PackageSpec(name="Metalhead"))
	Pkg.add(Pkg.PackageSpec(name="Markdown"))
	Pkg.add(Pkg.PackageSpec(name="NPZ"))
	Pkg.add(Pkg.PackageSpec(name="Plots"))
	Pkg.add(Pkg.PackageSpec(name="ProgressLogging"))
	Pkg.add(Pkg.PackageSpec(name="Random"))
	Pkg.add(Pkg.PackageSpec(name="Statistics"))
	Pkg.add(Pkg.PackageSpec(name="Tables"))
	Pkg.add(Pkg.PackageSpec(name="Tar"))
	Pkg.add(Pkg.PackageSpec(name="PlutoUI"))
	Pkg.add(Pkg.PackageSpec(name="UnicodePlots"))

	using Colors
	using CSV
	using Flux
	using GZip
	using Graphs
	using GridGraphs
	using Images
	using InferOpt
	using JSON
	using LinearAlgebra
	using Markdown: MD, Admonition, Code
	using Metalhead
	using NPZ
	using Plots
	using ProgressLogging
	using Random
	using Statistics
	using Tables
	using Tar
	using PlutoUI
	using UnicodePlots
	Random.seed!(63)
end;

# ╔═╡ 3a84fd20-41fa-4156-9be5-a0371754b394
md"""
# Shortest paths on Warcraft maps
"""

# ╔═╡ ee87d357-318f-40f1-a82a-fe680286e6cd
md"""
In this notebook, we define learning pipelines for the Warcraft shortest path problem. 
We have a sub-dataset of Warcraft terrain images, corresponding black-box cost functions, and optionally the label shortest path solutions and cell costs. 
We want to learn the cost of the cells, using a neural network embedding, to predict good shortest paths on new test images.
More precisely, each point in our dataset consists in:
- an image of terrain ``I``.
- a black-box cost function ``c`` to evaluate any given path.
- a label shortest path ``P`` from the top-left to the bottom-right corners (optional). 
- the true cost of each cell of the grid (optional).
We can exploit the images to approximate the true cell costs, so that when considering a new test image of terrain, we predict a good shortest path from its top-left to its bottom-right corners.
The question is: how should we combine these features?
We use `InferOpt` to learn the appropriate costs.
"""

# ╔═╡ e279878d-9c8d-47c8-9453-3aee1118818b
md"""
**Utilities (hidden)**
"""

# ╔═╡ 8b7876e4-2f28-42f8-87a1-459b665cff30
md"""
Imports
"""

# ╔═╡ a0d14396-cb6a-4f35-977a-cf3b63b44d9e
md"""
TOC
"""

# ╔═╡ b5b0bb58-9e02-4551-a9ba-0ba0ffceb350
TableOfContents()

# ╔═╡ 2182d4d2-6506-4fd6-936f-0e7c30d73851
html"""
<script>
    const calculate_slide_positions = (/** @type {Event} */ e) => {
        const notebook_node = /** @type {HTMLElement?} */ (e.target)?.closest("pluto-editor")?.querySelector("pluto-notebook")
		console.log(e.target)
        if (!notebook_node) return []
        const height = window.innerHeight
        const headers = Array.from(notebook_node.querySelectorAll("pluto-output h1, pluto-output h2"))
        const pos = headers.map((el) => el.getBoundingClientRect())
        const edges = pos.map((rect) => rect.top + window.pageYOffset)
        edges.push(notebook_node.getBoundingClientRect().bottom + window.pageYOffset)
        const scrollPositions = headers.map((el, i) => {
            if (el.tagName == "H1") {
                // center vertically
                const slideHeight = edges[i + 1] - edges[i] - height
                return edges[i] - Math.max(0, (height - slideHeight) / 2)
            } else {
                // align to top
                return edges[i] - 20
            }
        })
        return scrollPositions
    }
    const go_previous_slide = (/** @type {Event} */ e) => {
        const positions = calculate_slide_positions(e)
        const pos = positions.reverse().find((y) => y < window.pageYOffset - 10)
        if (pos) window.scrollTo(window.pageXOffset, pos)
    }
    const go_next_slide = (/** @type {Event} */ e) => {
        const positions = calculate_slide_positions(e)
        const pos = positions.find((y) => y - 10 > window.pageYOffset)
        if (pos) window.scrollTo(window.pageXOffset, pos)
    }
	const left_button = document.querySelector(".changeslide.prev")
	const right_button = document.querySelector(".changeslide.next")
	left_button.addEventListener("click", go_previous_slide)
	right_button.addEventListener("click", go_next_slide)
</script>
"""

# ╔═╡ 1f0c5b88-f903-4a67-9581-b3a07c504d5c
md"""
Two columns
"""

# ╔═╡ 91122ec3-bad8-46f6-8c5e-7715163a60d5
begin
	struct TwoColumn{L, R}
	    left::L
	    right::R
		leftfrac::Int
		rightfrac::Int
	end
	
	function Base.show(io, mime::MIME"text/html", tc::TwoColumn)
		(; left, right, leftfrac, rightfrac) = tc
	    write(io, """<div style="display: flex;"><div style="flex: $(leftfrac)%;">""")
	    show(io, mime, left)
	    write(io, """</div><div style="flex: $(rightfrac)%;">""")
	    show(io, mime, right)
	    write(io, """</div></div>""")
	end
end

# ╔═╡ 86735dcf-de5b-4f32-8bf9-501e006f58d5
begin
	info(text; title="Info") = MD(Admonition("info", title, [text]))
	tip(text; title="Tip") = MD(Admonition("tip", title, [text]))
	warning(text; title="Warning") = MD(Admonition("warning", title, [text]))
	danger(text; title="Danger") = MD(Admonition("danger", title, [text]))
	hint(text; title="Hint") = MD(Admonition("hint", title, [text]))
	not_defined(var) = warning(md"You must give a value to $(Code(string(var)))."; title="Undefined variable")
	keep_working() = info(md"You're almost there."; title="Keep working!")
	correct() = tip(md"Well done."; title="Correct!")
end;

# ╔═╡ 94192d5b-c4e9-487f-a36d-0261d9e86801
md"""
## I - Dataset and plots
"""

# ╔═╡ 98eb10dd-a4a1-4c91-a0cd-dd1d1e6bc89a
md"""
We first give the path of the dataset folder:
"""

# ╔═╡ 8d2ac4c8-e94f-488e-a1fa-611b7b37fcea
decompressed_path = joinpath(".", "data")

# ╔═╡ 4e2a1703-5953-4901-b598-9c1a98a5fc2b
md"""
### a) Gridgraphs
"""

# ╔═╡ 6d1545af-9fd4-41b2-9a9b-b4472d6c360e
md"""For the purposes of this TP, we consider grid graphs, as implemented in [GridGraphs.jl](https://github.com/gdalle/GridGraphs.jl).
In such graphs, each vertex corresponds to a couple of coordinates ``(i, j)``, where ``1 \leq i \leq h`` and ``1 \leq j \leq w``.
"""

# ╔═╡ e2c4292f-f2e8-4465-b3e3-66be158cacb5
h, w = 12, 12;

# ╔═╡ bd7a9013-199a-4bec-a5a4-4165da61f3cc
g = GridGraph(exp.(rand(100, 100)))

# ╔═╡ c04157e6-52a9-4d2e-add8-680dc71e5aaa
md"""For convenience, `GridGraphs.jl` also provides custom functions to compute shortest paths efficiently. We use the Dijkstra implementation.
Let us see what those paths look like.
"""

# ╔═╡ 16cae90f-6a37-4240-8608-05f3d9ab7eb5
begin
	p = path_to_matrix(g, grid_dijkstra(g, 1, nv(g)));
	UnicodePlots.spy(p)
end

# ╔═╡ 3044c025-bfb4-4563-8563-42a783e625e2
md"""
### b) Dataset functions
"""

# ╔═╡ 6d21f759-f945-40fc-aaa3-7374470c4ef0
md"""
The first dataset function `read_dataset` is used to read the images, cell costs and shortest path labels stored in files of the dataset folder.
"""

# ╔═╡ 3c141dfd-b888-4cf2-8304-7282aabb5aef
begin 
	"""
	    read_dataset(decompressed_path::String, dtype::String="train")

	Read the dataset of type `dtype` at the `decompressed_path` location.
	The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
	They are returned separately, with proper axis permutation and image scaling to be consistent with 
	`Flux` embeddings.
	"""
	function read_dataset(decompressed_path::String, dtype::String="train")
	    # Open files
	    data_dir = joinpath(decompressed_path, "warcraft_shortest_path_oneskin", "12x12")
	    data_suffix = "maps"
	    terrain_images = npzread(joinpath(data_dir, dtype * "_" * data_suffix * ".npy"))
	    terrain_weights = npzread(joinpath(data_dir, dtype * "_vertex_weights.npy"))
	    terrain_labels = npzread(joinpath(data_dir, dtype * "_shortest_paths.npy"))
	    # Reshape for Flux
	    terrain_images = permutedims(terrain_images, (2, 3, 4, 1))
	    terrain_labels = permutedims(terrain_labels, (2, 3, 1))
	    terrain_weights = permutedims(terrain_weights, (2, 3, 1))
	    # Normalize images
	    terrain_images = Array{Float32}(terrain_images ./ 255)
	    println("Train images shape: ", size(terrain_images))
	    println("Train labels shape: ", size(terrain_labels))
	    println("Weights shape:", size(terrain_weights))
	    return terrain_images, terrain_labels, terrain_weights
	end
end

# ╔═╡ c18d4b8f-2ae1-4fde-877b-f53823a42ab1
md"""
Once the files are read, we want to give an adequate format to the dataset, so that we can easily load samples to train and test models. The function `create_dataset` therefore calls the previous `read_dataset` function: 
"""

# ╔═╡ 8c8bb6a1-12cd-4af3-b573-c22383bdcdfb
begin
	"""
	    create_dataset(decompressed_path::String, nb_samples::Int=10000)

	Create the dataset corresponding to the data located at `decompressed_path`, possibly sub-sampling `nb_samples` points.
	The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
	It is a `Vector` of tuples, each `Tuple` being a dataset point.
	"""
	function create_dataset(decompressed_path::String, nb_samples::Int=10000)
	    terrain_images, terrain_labels, terrain_weights = read_dataset(
	        decompressed_path, "train"
	    )
	    X = [
	        reshape(terrain_images[:, :, :, i], (size(terrain_images[:, :, :, i])..., 1)) for
	        i in 1:nb_samples
	    ]
	    Y = [terrain_labels[:, :, i] for i in 1:nb_samples]
	    WG = [(wg=GridGraph(terrain_weights[:, :, i]),) for i in 1:nb_samples]
	    return collect(zip(X, Y, WG))
	end
end

# ╔═╡ 4a9ed677-e294-4194-bf32-9580d1e47bda
md"""
Last, as usual in machine learning implementations, we split a dataset into train and test sets. The function `train_test_split` does the job:

"""

# ╔═╡ 0514cde6-b425-4fe7-ac1e-2678b64bbee5
begin
	"""
	    train_test_split(X::AbstractVector, train_percentage::Real=0.5)

	Split a dataset contained in `X` into train and test datasets.
	The proportion of the initial dataset kept in the train set is `train_percentage`.
	"""
	function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
	    N = length(X)
	    N_train = floor(Int, N * train_percentage)
	    N_test = N - N_train
	    train_ind, test_ind = 1:N_train, (N_train + 1):(N_train + N_test)
	    X_train, X_test = X[train_ind], X[test_ind]
	    return X_train, X_test
	end
end

# ╔═╡ caf02d68-3418-4a6a-ae25-eabbbc7cae3f
md"""
### c) Plot functions
"""

# ╔═╡ 61db4159-84cd-4e3d-bc1e-35b35022b4be
md"""
In the following cell, we define utility plot functions to have a glimpse at images, cell costs and paths. Their implementation is not at the core of this TP, they are thus hidden.
"""

# ╔═╡ 08ea0d7e-2ffe-4f2e-bd8c-f15f9af0f35b
begin 
		"""
	    convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
	Convert `image` to the proper data format to enable plots in Julia.
	"""
	function convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
	    new_img = Array{RGB{N0f8},2}(undef, size(image)[1], size(image)[2])
	    for i = 1:size(image)[1]
	        for j = 1:size(image)[2]
	            new_img[i,j] = RGB{N0f8}(image[i,j,1], image[i,j,2], image[i,j,3])
	        end
	    end
	    return new_img
	end
	
	"""
	    plot_image_and_path(im::Array{RGB{N0f8}, 2}, zero_one_path::Matrix{UInt8})
	Plot the image `im` and the path `zero_one_path` on the same Figure.
	"""
	function plot_image_and_path(im::Array{RGB{N0f8}, 2}, zero_one_path::Matrix{UInt8})
	    p1 = plot(im, title = "Terrain map", ticks = nothing, border = nothing)
	    p2 = plot(Gray.(zero_one_path), title = "Path", ticks = nothing, border = nothing)
	    plot(p1, p2, layout = (1, 2))
	end
	
	"""
		plot_image_weights_path(;im, weights, path)
	Plot the image `im`, the weights `weights`, and the path `path` on the same Figure.
	"""
	function plot_image_weights_path(;im, weights, path)
		img = convert_image_for_plot(im)
	    p1 = Plots.plot(
	        img;
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(1000, 1000),
			title = "Terrain image"
	    )
	    p2 = Plots.heatmap(
			weights;
			yflip=true,
			aspect_ratio=:equal,
			framestyle=:none,
			padding=(0., 0.),
			size=(1000, 1000),
			legend = false,
			title = "Weights"
		)
	    p3 = Plots.plot(
	        Gray.(path .* 0.7);
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(1000, 1000),
			title = "Path"
	    )
	    plot(p1, p2, p3, layout = (1, 3), size = (3000, 1000))
	end
	
	"""
	    plot_loss_and_gap(losses::Matrix{Float64}, gaps::Matrix{Float64},  options::NamedTuple; filepath=nothing)
	
	Plot the train and test losses, as well as the train and test gaps computed over epochs.
	"""
	function plot_loss_and_gap(losses::Matrix{Float64}, gaps::Matrix{Float64},  options::NamedTuple; filepath=nothing)
	    x = collect(1:options.nb_epochs)
	    p1 = plot(x, losses, title = "Loss", xlabel = "epochs", ylabel = "loss", label = ["train" "test"])
	    p2 = plot(x, gaps, title = "Gap", xlabel = "epochs", ylabel = "ratio", label = ["train" "test"])
	    pl = plot(p1, p2, layout = (1, 2))
	    isnothing(filepath) || Plots.savefig(pl, filepath)
	    return pl
	end
	
	"""
	    plot_weights_path(;weights, path)
	Plot both the cell costs and path on the same colormap Figure.
	"""
	function plot_weights_path(;weights, path, weight_title="Weights", path_title="Path")
	    p1 = Plots.heatmap(
		        weights;
		        yflip=true,
		        aspect_ratio=:equal,
		        framestyle=:none,
		        padding=(0., 0.),
		        size=(500, 500),
				legend = false,
				title = weight_title
		)
	    p2 = Plots.plot(
	        Gray.(path .* 0.7);
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(500, 500),
			title = path_title
	    )
	    plot(p1, p2, layout = (1, 2), size = (1000, 500))
	end
	
	function plot_map(map_matrix::Array{<:Real,3}; filepath=nothing)
	    img = convert_image_for_plot(map_matrix)
	    pl = Plots.plot(
	        img;
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(500, 500)
	    )
	    isnothing(filepath) || Plots.savefig(pl, filepath)
	    return pl
	end
	
	function plot_weights(weights::Matrix{<:Real}; filepath=nothing)
	    pl = Plots.heatmap(
	        weights;
	        yflip=true,
	        aspect_ratio=:equal,
	        framestyle=:none,
	        padding=(0., 0.),
	        size=(500, 500)
	    )
	    isnothing(filepath) || Plots.savefig(pl, filepath)
	    return pl
	end
	
	function plot_path(path::Matrix{<:Integer}; filepath=nothing)
	    pl = Plots.plot(
	        Gray.(path .* 0.7);
	        aspect_ratio=:equal,
	        framestyle=:none,
	        size=(500, 500)
	    )
	    isnothing(filepath) || Plots.savefig(pl, filepath)
	    return pl
	end
end

# ╔═╡ d58098e8-bba5-445c-b1c3-bfb597789916
md"""
### d) Import and explore the dataset
"""

# ╔═╡ a0644bb9-bf62-46aa-958e-aeeaaba3482e
md"""
Once we have both defined the functions to read and create a dataset, and to visualize it, we want to have a look at images and paths. Before that, we set the size of the dataset, as well as the train proportion: 
"""

# ╔═╡ eaf0cf1f-a7be-4399-86cc-66c131a57e44
nb_samples, train_prop = 100, 0.8;

# ╔═╡ 2470f5ab-64d6-49d5-9816-0c958714ca73
info(md"We focus only on $nb_samples dataset points, and use a $(trunc(Int, train_prop*100))% / $(trunc(Int, 100 - train_prop*100))% train/test split.")

# ╔═╡ 73bb8b94-a45f-4dbb-a4f6-1f25ad8f194c
begin
	dataset = create_dataset(decompressed_path, nb_samples)
	train_dataset, test_dataset = train_test_split(dataset, train_prop);
end;

# ╔═╡ c9a05a6e-90c3-465d-896c-74bbb429f66a
md"""
We can have a glimpse at a dataset image as follows:
"""

# ╔═╡ fd83cbae-638e-49d7-88da-588fe055c963
md"""
``n =`` $(@bind n Slider(1:length(dataset); default=1, show_value=true))
"""

# ╔═╡ fe3d8a72-f68b-4162-b5f2-cc168e80a3c6
begin
	x, y, kwargs = dataset[n]
	plot_map(dropdims(x; dims=4))
end

# ╔═╡ 3ca72cd5-58f8-47e1-88ca-cd115b181e74
plot_weights_path(weights = kwargs.wg.weights, path =y)

# ╔═╡ 253e9920-8bfb-47ba-ad7d-b2ed3ebb5fa7
dataset[1][3]

# ╔═╡ fa62a7b3-8f17-42a3-8428-b2ac7eae737a
md"""
## II - Combinatorial functions
"""

# ╔═╡ 0f299cf1-f729-4999-af9d-4b39730100d8
md"""
We focus on additional optimization functions to define the combinatorial layer of our pipelines.
"""

# ╔═╡ e59b06d9-bc20-4d70-8940-5f0a53389738
md"""
### a) Recap on the shortest path problem
"""

# ╔═╡ 75fd015c-335a-481c-b2c5-4b33ca1a186a
md"""
Let $D = (V, A)$ be a digraph, $(c_a)_{a \in A}$ the cost associated to the arcs of the digraph, and $(o, d) \in V^2$ the origin and destination nodes. The problem we consider is the following:

**Shortest path problem:** Find an elementary path $P$ from node $o$ to node $d$ in the digraph $D$ with minimum cost $c(P) = \sum_{a \in P} c_a$.
"""

# ╔═╡ 7b653840-6292-4e6b-a6d6-91aadca3f6d4
md"""
!!! danger "Question"
	When the cost function is non-negative, which algorithm can we use ?
"""

# ╔═╡ 6b4f778a-e96c-4d0e-ac5e-3ecbb6d24df1
md"""
!!! info "Answer"
	Dijkstra algorithm
"""

# ╔═╡ 487eb4f1-cd50-47a7-8d61-b141c1b272f0
md"""
!!! danger "Question" 
	In the case the graph contains no absorbing cycle, which algorithm can we use ? 	On which principle is it based ?
"""

# ╔═╡ 91bd6598-5d97-4487-81aa-ec77eed0b53f
md"""
!!! info "Answer"
	Ford-Bellman algorithm （Dynamic programming）

	The subpath of an optimal path is still optimal
"""

# ╔═╡ 654066dc-98fe-4c3b-92a9-d09efdfc8080
md"""
In the following, we will perturb or regularize the output of a neural network to define the candidate cell costs to predict shortest paths. We therefore need to deal with possibly negative costs. 

!!! danger "Question"
	In the general case, can we fix the maximum length of a feasible solution of the shortest path problem ? How ? Can we derive a dynamic programming algorithm based on this ?
"""

# ╔═╡ 02367dbb-b7df-49d4-951f-6a0a701e958c
md"""
!!! info "Answer"
	the length between the top-left corner and the bottom-right corner
"""

# ╔═╡ f18ad74f-ef8b-4c70-8af3-e6dac8305dd0
begin
	
"""
    grid_bellman_ford_warcraft(g, s, d, length_max)

Apply the Bellman-Ford algorithm on an `GridGraph` `g`, and return a `ShortestPathTree` with source `s` and destination `d`,
among the paths having length smaller than `length_max`.
"""
function grid_bellman_ford_warcraft(g::GridGraph{T,R,W,A}, s::Integer, d::Integer, length_max::Int = nv(g)) where {T,R,W,A}
    # Init storage
    parents = zeros(T, nv(g), length_max+1)
    dists = Matrix{Union{Nothing,R}}(undef, nv(g), length_max+1)
    fill!(dists, Inf)
    # Add source
    dists[s,1] = zero(R)
    # Main loop
    for k in 1:length_max
        for v in vertices(g)
            for u in inneighbors(g, v)
                d_u = dists[u, k]
                if !isinf(d_u)
                    d_v = dists[v, k+1]
                    d_v_through_u = d_u + GridGraphs.vertex_weight(g, v)
                    if isinf(d_v) || (d_v_through_u < d_v)
                        dists[v, k+1] = d_v_through_u
                        parents[v, k+1] = u
                    end
                end
            end
        end
    end
    # Get length of the shortest path
    k_short = argmin(dists[d,:])
    if isinf(dists[d, k_short])
        println("No shortest path with less than $length_max arcs")
        return T[]
    end
    # Deduce the path
    v = d
    path = [v]
    k = k_short
    while v != s
        v = parents[v, k]
        if v == zero(T)
            return T[]
        else
            pushfirst!(path, v)
            k = k-1
        end
    end
    return path
end
end

# ╔═╡ dc359052-19d9-4f29-903c-7eb9b210cbcd
md"""
###  b) From shortest path to generic maximizer
"""

# ╔═╡ b93009a7-533f-4c5a-a4f5-4c1d88cc1be4
md"""
Now that we have defined and implemented an algorithm to deal with the shortest path problem, we wrap it in a maximizer function to match the generic framework of structured prediction.
"""

# ╔═╡ 9153d21d-709a-4619-92cc-82269e167c0c
begin
		"""
	    true_maximizer(θ::AbstractMatrix{R}; kwargs...) where {R<:Real}
	Compute the shortest path from top-left corner to down-right corner on a gridgraph of the size of `θ` as an argmax.
	The weights of the arcs are given by the opposite of the values of `θ` related 
	to their destination nodes. We use GridGraphs, implemented 
	in [GridGraphs.jl](https://github.com/gdalle/GridGraphs.jl).
	"""
	function true_maximizer(θ::AbstractMatrix{R}; kwargs...) where {R<:Real}
	    g = GridGraph(-θ)
	    path = grid_bellman_ford_warcraft(g, 1, nv(g))
	    #path = grid_dijkstra(g, 1, nv(g))
	    y = path_to_matrix(g, path)
	    return y
	end
end

# ╔═╡ 76d4caa4-a10c-4247-a624-b6bfa5a743bc
md"""
!!! info "The maximizer function will depend on the pipeline"
	Note that we use the function `grid_dijkstra` already implemented in the `GridGraphs.jl` package when we deal with non-negative cell costs. In the following, we will use either Dijkstra or Ford-Bellman algorithm depending on the learning pipeline. You will have to modify the `true_maximizer` function depending on the experience you do.
"""

# ╔═╡ 91ec470d-f2b5-41c1-a50f-fc337995c73f
md"""
## III - Learning functions
"""

# ╔═╡ f899c053-335f-46e9-bfde-536f642700a1
md"""
### a) Convolutional neural network: predictor for the cost vector
"""

# ╔═╡ 6466157f-3956-45b9-981f-77592116170d
md"""
We implement several elementary functions to define our machine learning predictor for the cell costs.
"""

# ╔═╡ 211fc3c5-a48a-41e8-a506-990a229026fc
begin
	"""
    average_tensor(x)

Average the tensor `x` along its third axis.
"""
function average_tensor(x)
    return sum(x, dims = [3])/size(x)[3]
end
end

# ╔═╡ 7b8b659c-9c7f-402d-aa7b-63c17179560e
begin 
	"""
    neg_exponential_tensor(x)

Compute minus exponential element-wise on tensor `x`.
"""
function neg_exponential_tensor(x)
    return -exp.(x)
end
end

# ╔═╡ e392008f-1a92-4937-8d8e-820211e44422
begin
	"""
    squeeze_last_dims(x)

Squeeze two last dimensions on tensor `x`.
"""
function squeeze_last_dims(x)
    return reshape(x, size(x)[1], size(x)[2])
end
end

# ╔═╡ 8f23f8cc-6393-4b11-9966-6af67c6ecd40
md"""
!!! info "CNN as predictor"
	The following function defines the convolutional neural network we will use as cell costs predictor.
"""

# ╔═╡ 51a44f11-646c-4f1a-916e-6c83750f8f20
begin
	"""
    create_warcraft_embedding()

Create and return a `Flux.Chain` embedding for the Warcraft terrains, inspired by [differentiation of blackbox combinatorial solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py).

The embedding is made as follows:
    1) The first 5 layers of ResNet18 (convolution, batch normalization, relu, maxpooling and first resnet block).
    2) An adaptive maxpooling layer to get a (12x12x64) tensor per input image.
    3) An average over the third axis (of size 64) to get a (12x12x1) tensor per input image.
    4) The element-wize [`neg_exponential_tensor`](@ref) function to get cell weights of proper sign to apply shortest path algorithms.
    4) A squeeze function to forget the two last dimensions. 
"""
function create_warcraft_embedding()
    resnet18 = ResNet(18, pretrain = false, nclasses = 1)
    model_embedding = Chain(
		resnet18.layers[1][1:4], 
        AdaptiveMaxPool((12,12)), 
        average_tensor, 
        neg_exponential_tensor, 
        squeeze_last_dims,
    )
    return model_embedding
end
end

# ╔═╡ d793acb0-fd30-48ba-8300-dff9caac536a
md"""
We can build the encoder in this way:
"""

# ╔═╡ d9f5281b-f34b-485c-a781-804b8472e38c
create_warcraft_embedding()

# ╔═╡ 9782f5fb-7e4b-4d8a-a77a-e4f5b9a71ab5
md"""
### b) Loss and gap utility functions
"""

# ╔═╡ 596734af-cf81-43c9-a525-7ea88a209a53
md"""
In the cell below, we define the `cost` function seen as black-box to evaluate the cost of a given path on the grid.
"""

# ╔═╡ 0ae90d3d-c718-44b2-81b5-25ce43f42988
cost(y; c_true, kwargs...) = dot(y, c_true)

# ╔═╡ 201ec4fd-01b1-49c4-a104-3d619ffb447b
md"""
The following cell defines the scaled half square norm function and its gradient.
"""

# ╔═╡ 8b544491-b892-499f-8146-e7d1f02aaac1
begin
	scaled_half_square_norm(x::AbstractArray{<:Real}, ϵ::R = 25.) where {R<:Real} = ϵ*sum(abs2, x) / 2
	
	grad_scaled_half_square_norm(x::AbstractArray{<:Real}, ϵ::R = 25.) where {R<:Real} = ϵ*identity(x)
end

# ╔═╡ 6a482757-8a04-4724-a3d2-33577748bd4e
md"""
During training, we want to evaluate the quality of the predicted paths, both on the train and test datasets. We define the shortest path cost ratio between a candidate shortest path $\hat{y}$ and the label shortest path $y$ as: $r(\hat{y},y) = c(\hat{y}) / c(y)$.
"""

# ╔═╡ c89f17b8-fccb-4d62-a0b7-a84bbfa543f7
md"""
!!! danger "Question"
	What is the link in our problem between the shortest path cost ratio and the gap of a given solution with respect to the optimal solution ?
"""

# ╔═╡ 4d7070f2-b953-4555-b5d5-8fd0cee28b68
md"""
!!! info "Answer"
	When the the shortest path cost ratio equals to 1, the gap will be 0. 
	When the raitio is getting closer to 1, the gap is getting smaller.
"""

# ╔═╡ 26c71a94-5b30-424f-8242-c6510d41bb52
begin 
	"""
    shortest_path_cost_ratio(model, x, y, kwargs)
Compute the ratio between the cost of the solution given by the `model` cell costs and the cost of the true solution.
We evaluate both the shortest path with respect to the weights given by `model(x)` and the labelled shortest path `y`
using the true cell costs stored in `kwargs.wg.weights`. 
This ratio is by definition greater than one. The closer it is to one, the better is the solution given by the current 
weights of `model`. We thus track this metric during training.
"""
function shortest_path_cost_ratio(model, x, y, kwargs)
    true_weights = kwargs.wg.weights
    θ_computed = model(x)
    shortest_path_computed = true_maximizer(θ_computed)
    return dot(true_weights, shortest_path_computed)/dot(y, true_weights)
end
end

# ╔═╡ b25f438f-832c-4717-bb73-acbb22aec384
md"""
The two following functions extend the shortest path cost ratio to a batch and a dataset.

"""

# ╔═╡ dd1791a8-fa59-4a36-8794-fccdcd7c912a
begin
"""
    shortest_path_cost_ratio(model, batch)
Compute the average cost ratio between computed and true shorest paths over `batch`. 
"""
function shortest_path_cost_ratio(model, batch)
    return sum(shortest_path_cost_ratio(model, item[1], item[2], item[3]) for item in batch)/length(batch)
end
end

# ╔═╡ 633e9fea-fba3-4fe6-bd45-d19f89cb1808
begin
	"""
    shortest_path_cost_ratio(;model, dataset)
Compute the average cost ratio between computed and true shorest paths over `dataset`. 
"""
function shortest_path_cost_ratio(;model, dataset)
    return sum(shortest_path_cost_ratio(model, batch) for batch in dataset)/length(dataset)
end
end

# ╔═╡ 8c8b514e-8478-4b2b-b062-56832115c670
md"""
### c) Main training function:
"""

# ╔═╡ 93dd97e6-0d37-4d94-a3f6-c63dc856fa66
md"""
We now consider the generic learning function. We want to minimize a given `flux_loss` over the `train_dataset`, by updating the parameters of `encoder`. We do so using `Flux.jl` package which contains utility functions to backpropagate in a stochastic gradient descent framework. We also track the loss and cost ratio metrics both on the train and test sets. The hyper-parameters are stored in the `options` tuple. 
"""

# ╔═╡ d35f0e8b-6634-412c-b5f3-ffd11246276c
md"""
The following block defines the generic learning function.
"""

# ╔═╡ a6a56523-90c9-40d2-9b68-26e20c1a5527
begin 
	"""
    train_function!(;encoder, flux_loss, train_dataset, test_dataset, options::NamedTuple)
Train `encoder` model over `train_dataset` and test on `test_dataset` by minimizing `flux_loss` loss. 
This training involves differentiation through argmax with perturbed maximizers, using [InferOpt](https://github.com/axelparmentier/InferOpt.jl) package.
The task is to learn the best parameters for the `encoder`, so that when solving the shortest path problem with its output cell costs, the 
given solution is close to the labelled shortest path corresponding to the input Warcraft terrain image.
Hyperparameters are passed with `options`. During training, the average train and test losses are stored, as well as the average 
cost ratio computed with [`shortest_path_cost_ratio`](@ref) both on the train and test datasets.
"""
function train_function!(;encoder, flux_loss, train_dataset, test_dataset, options::NamedTuple)
    # Store the train loss
    losses = Matrix{Float64}(undef, options.nb_epochs, 2)
    cost_ratios = Matrix{Float64}(undef, options.nb_epochs, 2)
    # Optimizer
    opt = ADAM(options.lr_start)
    # model parameters
    par = Flux.params(encoder)
    # Train loop
    @progress "Training epoch: " for epoch in 1:options.nb_epochs
        for batch in train_dataset
            batch_loss = 0
            gs = gradient(par) do
                batch_loss = flux_loss(batch)
            end
            losses[epoch, 1] += batch_loss
            Flux.update!(opt, par, gs)
        end
        losses[epoch, 1] = losses[epoch, 1]/(options.nb_samples*0.8)
        losses[epoch, 2] = sum([flux_loss(batch) for batch in test_dataset])/(options.nb_samples*0.2)
        cost_ratios[epoch, 1] = shortest_path_cost_ratio(model = encoder, dataset = train_dataset)
        cost_ratios[epoch, 2] = shortest_path_cost_ratio(model = encoder, dataset = test_dataset)
    end
     return losses, cost_ratios
end
	
end

# ╔═╡ 920d94cd-bfb5-4c02-baa3-f346d5c95e2e
md"""
## IV - Pipelines
"""

# ╔═╡ 658bd4b9-ee97-4b81-9337-ee6d1ccdf7bb
md"""
!!! info "Preliminary remark"
	Here come the specific learning experiments. The following code cells will have to be modified to deal with different settings (as well as the definition of the `true_maximizer` function and the one of the half square norm regularization above).
"""

# ╔═╡ f1b50452-4e8c-4393-b112-7a4cfb3b7fb4
md"""
As you know, the solution of a linear program is not differentiable with respect to its cost vector. Therefore, we need additional tricks to be able to update the parameters of the CNN defined by `create_warcraft_embedding`. Two points of view can be adopted: perturb or regularize the maximization problem. They can be unified when introducing probabilistic combinatorial layers, detailed in this [paper](https://arxiv.org/pdf/2207.13513.pdf). They are used in two different frameworks:

- Learning by imitation when we have target shortest path examples in the dataset.
- Learning by experience when we only have access to the images and to a black-box cost function to evaluate any candidate path.

In this section, we explore different combinatorial layers, as well as the learning by imitation and learning by experience settings.
"""

# ╔═╡ 9a670af7-cc20-446d-bf22-4e833cc9d854
md"""
### 1) Learning by imitation with additive perturbation
"""

# ╔═╡ f6949520-d10f-4bae-8d41-2ec880ac7484
md"""
In this framework, we use a perturbed maximizer to learn the parameters of the neural network. Given a maximization problem $y^*(\theta) := \operatorname{argmax}_{y \in \mathcal{C}} \langle y, \theta \rangle$, we define the additive perturbed maximization as:
"""

# ╔═╡ 9bef7690-0db3-4ba5-be77-0933ceb6215e
md"""
$y^+_\epsilon (\theta) := \mathbb{E}_{Z}\big[ \operatorname{argmax}_{y \in \mathcal{C}} \langle y, \theta + \epsilon Z \rangle \big]$ 
"""

# ╔═╡ c872d563-421c-4581-a8fa-a02cee58bc85
md"""
$F^+_\epsilon (\theta) := \mathbb{E}_{Z}\big[ \operatorname{max}_{y \in \mathcal{C}} \langle y, \theta + \epsilon Z \rangle \big]$ 
"""

# ╔═╡ 4d50d263-eca0-48ad-b32c-9b767cc57914
md"""
!!! danger "Question"
	From your homework, what can you say about $F^+_\epsilon (\theta)$ and $y^+_\epsilon (\theta)$ ? What are their properties ? 
"""

# ╔═╡ 87f52425-dd46-4233-82d1-1203f66a4a35
md"""
!!! info "Answer"
	We found that $y^+_\epsilon (\theta)$ is the gradient of $F^+_\epsilon (\theta)$.
	
	We know that they are convex
"""

# ╔═╡ e4b13e58-2d54-47df-b865-95ae2946756a
md"""
Let $\Omega_\epsilon^+$ be the Fenchel conjugate of $F^+_\epsilon (\theta)$, we can define the natural Fenchel-Young loss as:

"""

# ╔═╡ 9c05cae5-af20-4f63-99c9-86032358ffd3
md"""
$L_\epsilon^+ (\theta, y) := F^+_{\epsilon} (\theta) + \Omega_{\epsilon}^+ (y) - \langle \theta, y \rangle$
"""

# ╔═╡ d2e5f60d-199a-41f5-ba5d-d21ab2030fb8
md"""
!!! danger "Question"
	What are the properties of $L_\epsilon^+ (\theta, y)$ ?
"""

# ╔═╡ 61514b98-a3fb-47fc-ba73-60baf8c80a87
md"""
!!! info "Answer"
	First is that $L_\epsilon^+ (\theta, y)$ is convex to $\theta$
	
	Second is that $L_\epsilon^+ (\theta, y) \geq 0$

	Third is that $L_\epsilon^+ (\theta, y) = 0 $ when $\theta$ and $y$ are Fenchel conjugate
"""

# ╔═╡ c10844d0-8328-42db-b49c-23713b9d88c6
md"""
!!! danger "Todo"
	Based on part I-III, run the end of the notebook to train and test the pipeline learning by imitation with additive perturbation and Fenchel-Young loss.
"""

# ╔═╡ 9a9b3942-72f2-4c9e-88a5-af927634468c
md"""
### 2) Learning by imitation with multiplicative perturbation
"""

# ╔═╡ 1ff198ea-afd5-4acc-bb67-019051ff149b
md"""
We introduce a variant of the additive pertubation defined above, which is simply based on an element-wise product $\odot$:
"""

# ╔═╡ 44ece9ce-f9f1-46f3-90c6-cb0502c92c67
md"""
${y}_\epsilon^\odot (\theta) := \mathbb{E}_Z \bigg[\operatorname{argmax}_{y \in \mathcal{C}} \langle \theta \odot e^{\epsilon Z - \epsilon^2 \mathbf{1} / 2},  y \rangle \bigg]$
"""

# ╔═╡ 5d8d34bb-c207-40fc-ab10-c579e7e2d04c
md"""
!!! danger "Question"
	What is the advantage of this perturbation compared with the additive one in terms of combinatorial problem ? Which algorithm can we use to compute shortest paths ?
"""

# ╔═╡ 07d880f3-33b7-4397-9ced-a135d9f0de22
md"""
!!! info "Answer"
	Cost will be non-negative
	
	Dijkstra algorithm
"""

# ╔═╡ 43d68541-84a5-4a63-9d8f-43783cc27ccc
md"""
We omit the details of the loss derivations and concentrate on implementation.

!!! danger "Todo"
	Modify the code below to learn by imitation with multiplicative perturbation using [`InferOpt.jl`](https://axelparmentier.github.io/InferOpt.jl/dev/) package.

"""

# ╔═╡ 90a47e0b-b911-4728-80b5-6ed74607833d
md"""
### 3) Learning by experience with multiplicative perturbation
"""

# ╔═╡ 5d79b8c1-beea-4ff9-9830-0f5e1c4ef29f
md"""
When we restrict the train dataset to images $I$ and black-box cost functions $c$, we can not learn by imitation. We can instead derive a surrogate version of the regret that is differentiable. 

!!! info "Reading"
	Read Section 4.1 of this [paper](https://arxiv.org/pdf/2207.13513.pdf).

!!! danger "Todo"
	Modify the code below to learn by experience using a multiplicative perturbation and the black-box cost function.
"""

# ╔═╡ a5bfd185-aa77-4e46-a6b6-d43c4785a7fa
md"""
### 4) Learning by experience with half square norm regularization (bonus). 
"""

# ╔═╡ a7b6ecbd-1407-44dd-809e-33311970af12
md"""
For the moment, we have only considered perturbations to derive meaningful gradient information. We now focus on a half square norm regularization.

!!! danger "Todo"
	Based on the functions `scaled_half_square_norm` and `grad_scaled_half_square_norm`, use the `RegularizedGeneric` implementation of [`InferOpt.jl`](https://axelparmentier.github.io/InferOpt.jl/dev/algorithms/) to learn by experience. Modify the cells below to do so.
"""

# ╔═╡ b389a6a0-dc8e-4c6f-8a82-4f8878ffe879
md"""
#### a) Hyperparameters
"""

# ╔═╡ e0e97839-884a-49ed-bee4-f1f2ace5f5e0
md"""
We first define the hyper-parameters for the learning process. They include:
- The regularization size $\epsilon$.
- The number of samples drawn for the approximation of the expectation $M$.
- The number of learning epochs `nb_epochs`.
- The number of samples in the sub-dataset (train + test) `nb_samples`.
- The batch size for the stochastic gradient descent `batch_size`.
- The starting learning rate for ADAM optimizer `lr_start`.
"""

# ╔═╡ bcdd60b8-e0d8-4a70-88d6-725269447c9b
begin 
	ϵ = 0.05
	M = 20
	nb_epochs = 50
	batch_size = 80
	lr_start = 0.001
	options = (ϵ=ϵ, M=M, nb_epochs=nb_epochs, nb_samples=nb_samples, batch_size = batch_size, lr_start = lr_start)
end

# ╔═╡ 9de99f4a-9970-4be1-9e16-e64ed4e10277
md"""
#### b) Specific pipeline
"""

# ╔═╡ 518e7077-d61b-4f60-987f-d556e3eb1d0d
md"""
!!! info "What is a pipeline ?"
	This portion of code is the crucial part to define the learning pipeline. It contains: 
	- an encoder, the machine learning predictor, in our case a CNN.
	- a maximizer possibly applied to the output of the encoder before computing the loss.
	- a differentiable loss to evaluate the quality of the output of the pipeline.
	
	Its definition depends on the learning setting we consider.
"""

# ╔═╡ 1337513f-995f-4dfa-827d-797a5d2e5e1a
begin
	pipeline = (
	    encoder=create_warcraft_embedding(),
	    maximizer=identity,
		loss = FenchelYoungLoss(PerturbedAdditive(true_maximizer; ε=options.ϵ, nb_samples=options.M)),
		#loss=SPOPlusLoss(true_maximizer)
	)
end

# ╔═╡ f5e789b2-a62e-4818-90c3-76f39ea11aaa
md"""
#### c) Flux loss definition
"""

# ╔═╡ efa7736c-22c0-410e-94da-1df315f22bbf
md"""
From the generic definition of the pipeline we define a loss function compatible with `Flux.jl` package. Its definition depends on the learning setting we consider.
"""

# ╔═╡ e9df3afb-fa04-440f-9664-3496da85696b
begin
	(; encoder, maximizer, loss) = pipeline
	#flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)); c_true = #kwargs.wg.weights, fw_kwargs = (max_iteration=50,))
	flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)), y; fw_kwargs = (max_iteration=50,))
	#flux_loss_point(x, y, kwargs) = loss(maximizer(encoder(x)), -kwargs.wg.weights)
	flux_loss_batch(batch) = sum(flux_loss_point(item[1], item[2], item[3]) for item in batch)
end

# ╔═╡ 58b7267d-491d-40f0-b4ba-27ed0c9cc855
md"""
#### d) Apply the learning function
"""

# ╔═╡ ac76b646-7c28-4384-9f04-5e4de5df154f
md"""
Given the specific pipeline and loss, we can apply our generic train function to update the weights of the CNN predictor.
"""

# ╔═╡ 83a14158-33d1-4f16-85e1-2726c8fbbdfc
begin
	Losses, Cost_ratios = train_function!(;
	    encoder=encoder,
	    flux_loss = flux_loss_batch,
	    train_dataset=Flux.DataLoader(train_dataset; batchsize=batch_size),
	    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
	    options=options,
	)
	Gaps = (Cost_ratios .- 1) .* 100
end;

# ╔═╡ 4b31dca2-0195-4899-8a3a-e9772fabf495
md"""
#### e) Plot results
"""

# ╔═╡ 79e0deab-1e36-4863-ad10-187ed8555c72
md"""
Loss and gap over epochs, train and test datasets.
"""

# ╔═╡ 66d385ba-9c6e-4378-b4e0-e54a4df346a5
begin
	plot_loss_and_gap(Losses, Gaps, options)
end

# ╔═╡ 414cdd7a-fa93-4554-8b76-8f3637b08406
md"""
!!! danger "Question"
	Comment the loss and gap profiles. What can you deduce ?
"""

# ╔═╡ 6bf478d1-207b-4af5-b006-8a07002c8c4e
md"""
!!! info "Answer"
	The loss and Gap have a similar downward trend. the Fenchel-Young loss is a good reflection of the error between the predicted and true results.
"""

# ╔═╡ db799fa2-0e48-43ee-9ee1-80ff8d2e5de7
md"""
To assess performance, we can compare the true and predicted paths.
"""

# ╔═╡ eb3a6009-e181-443c-bb77-021e867030e4
md"""
!!! info "Visualize the model performance"
	We now want to see the effect of the learning process on the predicted costs and shortest paths.
"""

# ╔═╡ 521f5ffa-2c22-44c5-8bdb-67410431ca2e
begin
	container = []
	for (x, y, k) in test_dataset
		θp = encoder(x)
		yp = UInt8.(true_maximizer(θp))
		push!(container, (x, y, k, θp, yp))
	end
end

# ╔═╡ f9b35e98-347f-4ebd-a690-790c7b0e03d8
md"""
``j =`` $(@bind j Slider(1:length(test_dataset); default=1, show_value=true))
"""

# ╔═╡ 842bf89d-45eb-462d-ba74-ca260a8b177d
begin
	x_test, y_test, kwargs_test, θpred, ypred = container[j]
	plot_map(dropdims(x_test; dims=4))
end

# ╔═╡ 80fa8831-924f-4093-a89c-bf8fc440da6b
plot_weights_path(weights=kwargs_test.wg.weights, path=y_test, weight_title="True weights", path_title="Label shortest path")

# ╔═╡ 4a3630ca-c8dd-4e81-8ee2-bb0fc6b01a93
plot_weights_path(weights=-θpred, path=ypred, weight_title="Predicted weight", path_title="Predicted path")

# ╔═╡ 77353ccd-8788-4c68-9560-03f8c8100de1
md"""
#### Learning by imitation with multiplicative perturbation
"""

# ╔═╡ 28542b8c-0b1c-498a-9b59-2a008c86c532
md"""
!!! info "Comment"
	As the cost is non-negative here, we choose the dijkstra algorithm to generate our shortest route.
"""

# ╔═╡ a49c4e3a-f233-4d24-9ca3-2b513fe1def7
begin
	function true_maximizer_dij(θ::AbstractMatrix{R}; kwargs...) where {R<:Real}
	    g = GridGraph(-θ)
	    #path = grid_bellman_ford_warcraft(g, 1, nv(g))
	    path = grid_dijkstra(g, 1, nv(g))
	    y = path_to_matrix(g, path)
	    return y
	end
end

# ╔═╡ c5971140-9caa-4685-be6d-c2c01b724460
begin
	pipeline_2 = (
	    encoder_2=create_warcraft_embedding(),
	    maximizer_2=identity,
		loss_2 = FenchelYoungLoss(PerturbedMultiplicative(true_maximizer_dij; ε=options.ϵ, nb_samples=options.M)),
		#loss=SPOPlusLoss(true_maximizer)
	)
end

# ╔═╡ 2438af36-f51d-44a7-a314-383e70b51271
begin
	(; encoder_2, maximizer_2, loss_2) = pipeline_2
	flux_loss_point_2(x, y, kwargs) = loss_2(maximizer_2(encoder_2(x)), y; fw_kwargs = (max_iteration=50,))
	flux_loss_batch_2(batch) = sum(flux_loss_point_2(item[1], item[2], item[3]) for item in batch)
end

# ╔═╡ 4548d672-97ce-4b44-894c-3d75f693ed59
begin
	Losses_2, Cost_ratios_2 = train_function!(;
	    encoder=encoder_2,
	    flux_loss = flux_loss_batch_2,
	    train_dataset=Flux.DataLoader(train_dataset; batchsize=batch_size),
	    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
	    options=options,
	)
	Gaps_2 = (Cost_ratios_2 .- 1) .* 100
end;

# ╔═╡ d11d201a-989d-4b0f-a9e3-3611ad54f283
begin
	plot_loss_and_gap(Losses_2, Gaps_2, options)
end

# ╔═╡ 99705bd4-8f55-42b9-b3a8-3470ed153f04
md"""
#### Learning by experience with multiplicative perturbation
"""

# ╔═╡ 8dbfd2c8-139d-4c22-a35c-d48ee92472f8
options_3 = (ϵ=0.02, M=M, nb_epochs=nb_epochs, nb_samples=nb_samples, batch_size = batch_size, lr_start = lr_start)

# ╔═╡ 9000cf83-d1be-4c6c-b53e-f0f7e24eac60
begin
	perturbed_mult = PerturbedMultiplicative(true_maximizer_dij;ε=options_3.ϵ, nb_samples=options_3.M)
	regret_pert_mult = Pushforward(perturbed_mult, cost)
end

# ╔═╡ 7570cc23-8536-4648-8904-07bc776b1c94
begin
	pipeline_3 = (
	    encoder_3=create_warcraft_embedding(),
	    maximizer_3=identity,
		loss_3 = regret_pert_mult,
		#loss=SPOPlusLoss(true_maximizer)
	)
end

# ╔═╡ fb49b357-a334-4415-8c08-c936745918ae
md"""
!!! info "Comment"
	Here we do not have y in the expression of our loss function. 
"""

# ╔═╡ 7a233dd0-620f-4d4c-9501-a52891ea3de5
begin
	(; encoder_3, maximizer_3, loss_3) = pipeline_3
	flux_loss_point_3(x, y, kwargs) = loss_3(maximizer_3(encoder_3(x)); c_true = kwargs.wg.weights, fw_kwargs = (max_iteration=50,))
	flux_loss_batch_3(batch) = sum(flux_loss_point_3(item[1], item[2], item[3]) for item in batch)
end

# ╔═╡ 0d04e925-aa17-4753-a4f6-797b0742ff1d
begin
	Losses_3, Cost_ratios_3 = train_function!(;
	    encoder=encoder_3,
	    flux_loss = flux_loss_batch_3,
	    train_dataset=Flux.DataLoader(train_dataset; batchsize=batch_size),
	    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
	    options=options,
	)
	Gaps_3 = (Cost_ratios_3 .- 1) .* 100
end;

# ╔═╡ 89e01d48-bd57-40c1-88e2-aef7f2fa78c2
begin
	plot_loss_and_gap(Losses_3, Gaps_3, options_3)
end

# ╔═╡ 2d08e21e-34ed-4c52-b076-67fc39d63f80
md"""
#### Learning by experience with half square norm regularization
"""

# ╔═╡ 7e4e3906-f79a-4ecf-84da-f84c44c5476a
begin
	regularized = RegularizedGeneric(true_maximizer; omega=scaled_half_square_norm, omega_grad=grad_scaled_half_square_norm)
	regret_reg = Pushforward(regularized, cost)
end

# ╔═╡ 53c6e347-e4f3-4a75-9bf0-cb5cd3cdbe0b
begin
	pipeline_4 = (
	    encoder_4=create_warcraft_embedding(),
	    maximizer_4=identity,
		loss_4 = regret_reg,
		#loss=SPOPlusLoss(true_maximizer)
	)
end

# ╔═╡ e519133a-78b3-47ad-b375-3a5ec453b229
begin
	(; encoder_4, maximizer_4, loss_4) = pipeline_4
	flux_loss_point_4(x, y, kwargs) = loss_4(maximizer_4(encoder_4(x)); c_true = kwargs.wg.weights, fw_kwargs = (max_iteration=50,))
	flux_loss_batch_4(batch) = sum(flux_loss_point_4(item[1], item[2], item[3]) for item in batch)
end

# ╔═╡ b9e13f67-36a6-4a4f-afa9-5e0321cf702b
begin
	Losses_4, Cost_ratios_4 = train_function!(;
	    encoder=encoder_4,
	    flux_loss = flux_loss_batch_4,
	    train_dataset=Flux.DataLoader(train_dataset; batchsize=batch_size),
	    test_dataset = Flux.DataLoader(test_dataset; batchsize=length(test_dataset)),
	    options=options,
	)
	Gaps_4 = (Cost_ratios_4 .- 1) .* 100
end;

# ╔═╡ b1eebc9c-734b-4c2a-b08e-d4196b2f9cbd
begin
	plot_loss_and_gap(Losses_4, Gaps_4, options)
end

# ╔═╡ 70f1127d-64ba-4824-895d-252e71c8d4de
md"""
!!! info "Comment"
	We find that the performance of the method of learning by imitation is better than the method of learning by experience. It is normal as The method of learning by experience lacks one condition compared to the method of learning by imitation. Also, learning by experience is sensitive to the hyperparameters. But we can still do the gradient decsent to train the model to decrease the loss and the gap.
"""

# ╔═╡ Cell order:
# ╟─3a84fd20-41fa-4156-9be5-a0371754b394
# ╟─ee87d357-318f-40f1-a82a-fe680286e6cd
# ╟─e279878d-9c8d-47c8-9453-3aee1118818b
# ╟─8b7876e4-2f28-42f8-87a1-459b665cff30
# ╟─6160785a-d455-40b4-ab74-e61c46e31537
# ╟─a0d14396-cb6a-4f35-977a-cf3b63b44d9e
# ╟─b5b0bb58-9e02-4551-a9ba-0ba0ffceb350
# ╟─2182d4d2-6506-4fd6-936f-0e7c30d73851
# ╟─1f0c5b88-f903-4a67-9581-b3a07c504d5c
# ╟─91122ec3-bad8-46f6-8c5e-7715163a60d5
# ╟─86735dcf-de5b-4f32-8bf9-501e006f58d5
# ╟─94192d5b-c4e9-487f-a36d-0261d9e86801
# ╟─98eb10dd-a4a1-4c91-a0cd-dd1d1e6bc89a
# ╠═8d2ac4c8-e94f-488e-a1fa-611b7b37fcea
# ╟─4e2a1703-5953-4901-b598-9c1a98a5fc2b
# ╟─6d1545af-9fd4-41b2-9a9b-b4472d6c360e
# ╠═e2c4292f-f2e8-4465-b3e3-66be158cacb5
# ╠═bd7a9013-199a-4bec-a5a4-4165da61f3cc
# ╟─c04157e6-52a9-4d2e-add8-680dc71e5aaa
# ╠═16cae90f-6a37-4240-8608-05f3d9ab7eb5
# ╟─3044c025-bfb4-4563-8563-42a783e625e2
# ╟─6d21f759-f945-40fc-aaa3-7374470c4ef0
# ╟─3c141dfd-b888-4cf2-8304-7282aabb5aef
# ╟─c18d4b8f-2ae1-4fde-877b-f53823a42ab1
# ╟─8c8bb6a1-12cd-4af3-b573-c22383bdcdfb
# ╟─4a9ed677-e294-4194-bf32-9580d1e47bda
# ╟─0514cde6-b425-4fe7-ac1e-2678b64bbee5
# ╟─caf02d68-3418-4a6a-ae25-eabbbc7cae3f
# ╟─61db4159-84cd-4e3d-bc1e-35b35022b4be
# ╟─08ea0d7e-2ffe-4f2e-bd8c-f15f9af0f35b
# ╟─d58098e8-bba5-445c-b1c3-bfb597789916
# ╟─a0644bb9-bf62-46aa-958e-aeeaaba3482e
# ╟─2470f5ab-64d6-49d5-9816-0c958714ca73
# ╠═eaf0cf1f-a7be-4399-86cc-66c131a57e44
# ╠═73bb8b94-a45f-4dbb-a4f6-1f25ad8f194c
# ╟─c9a05a6e-90c3-465d-896c-74bbb429f66a
# ╟─fe3d8a72-f68b-4162-b5f2-cc168e80a3c6
# ╟─fd83cbae-638e-49d7-88da-588fe055c963
# ╠═3ca72cd5-58f8-47e1-88ca-cd115b181e74
# ╠═253e9920-8bfb-47ba-ad7d-b2ed3ebb5fa7
# ╟─fa62a7b3-8f17-42a3-8428-b2ac7eae737a
# ╟─0f299cf1-f729-4999-af9d-4b39730100d8
# ╟─e59b06d9-bc20-4d70-8940-5f0a53389738
# ╟─75fd015c-335a-481c-b2c5-4b33ca1a186a
# ╟─7b653840-6292-4e6b-a6d6-91aadca3f6d4
# ╟─6b4f778a-e96c-4d0e-ac5e-3ecbb6d24df1
# ╟─487eb4f1-cd50-47a7-8d61-b141c1b272f0
# ╟─91bd6598-5d97-4487-81aa-ec77eed0b53f
# ╟─654066dc-98fe-4c3b-92a9-d09efdfc8080
# ╟─02367dbb-b7df-49d4-951f-6a0a701e958c
# ╟─f18ad74f-ef8b-4c70-8af3-e6dac8305dd0
# ╟─dc359052-19d9-4f29-903c-7eb9b210cbcd
# ╟─b93009a7-533f-4c5a-a4f5-4c1d88cc1be4
# ╟─9153d21d-709a-4619-92cc-82269e167c0c
# ╟─76d4caa4-a10c-4247-a624-b6bfa5a743bc
# ╟─91ec470d-f2b5-41c1-a50f-fc337995c73f
# ╟─f899c053-335f-46e9-bfde-536f642700a1
# ╟─6466157f-3956-45b9-981f-77592116170d
# ╟─211fc3c5-a48a-41e8-a506-990a229026fc
# ╟─7b8b659c-9c7f-402d-aa7b-63c17179560e
# ╟─e392008f-1a92-4937-8d8e-820211e44422
# ╟─8f23f8cc-6393-4b11-9966-6af67c6ecd40
# ╟─51a44f11-646c-4f1a-916e-6c83750f8f20
# ╟─d793acb0-fd30-48ba-8300-dff9caac536a
# ╠═d9f5281b-f34b-485c-a781-804b8472e38c
# ╟─9782f5fb-7e4b-4d8a-a77a-e4f5b9a71ab5
# ╟─596734af-cf81-43c9-a525-7ea88a209a53
# ╠═0ae90d3d-c718-44b2-81b5-25ce43f42988
# ╟─201ec4fd-01b1-49c4-a104-3d619ffb447b
# ╠═8b544491-b892-499f-8146-e7d1f02aaac1
# ╟─6a482757-8a04-4724-a3d2-33577748bd4e
# ╟─c89f17b8-fccb-4d62-a0b7-a84bbfa543f7
# ╟─4d7070f2-b953-4555-b5d5-8fd0cee28b68
# ╟─26c71a94-5b30-424f-8242-c6510d41bb52
# ╟─b25f438f-832c-4717-bb73-acbb22aec384
# ╟─dd1791a8-fa59-4a36-8794-fccdcd7c912a
# ╟─633e9fea-fba3-4fe6-bd45-d19f89cb1808
# ╟─8c8b514e-8478-4b2b-b062-56832115c670
# ╟─93dd97e6-0d37-4d94-a3f6-c63dc856fa66
# ╟─d35f0e8b-6634-412c-b5f3-ffd11246276c
# ╟─a6a56523-90c9-40d2-9b68-26e20c1a5527
# ╟─920d94cd-bfb5-4c02-baa3-f346d5c95e2e
# ╟─658bd4b9-ee97-4b81-9337-ee6d1ccdf7bb
# ╟─f1b50452-4e8c-4393-b112-7a4cfb3b7fb4
# ╟─9a670af7-cc20-446d-bf22-4e833cc9d854
# ╟─f6949520-d10f-4bae-8d41-2ec880ac7484
# ╟─9bef7690-0db3-4ba5-be77-0933ceb6215e
# ╟─c872d563-421c-4581-a8fa-a02cee58bc85
# ╟─4d50d263-eca0-48ad-b32c-9b767cc57914
# ╟─87f52425-dd46-4233-82d1-1203f66a4a35
# ╟─e4b13e58-2d54-47df-b865-95ae2946756a
# ╟─9c05cae5-af20-4f63-99c9-86032358ffd3
# ╟─d2e5f60d-199a-41f5-ba5d-d21ab2030fb8
# ╟─61514b98-a3fb-47fc-ba73-60baf8c80a87
# ╟─c10844d0-8328-42db-b49c-23713b9d88c6
# ╟─9a9b3942-72f2-4c9e-88a5-af927634468c
# ╟─1ff198ea-afd5-4acc-bb67-019051ff149b
# ╟─44ece9ce-f9f1-46f3-90c6-cb0502c92c67
# ╟─5d8d34bb-c207-40fc-ab10-c579e7e2d04c
# ╟─07d880f3-33b7-4397-9ced-a135d9f0de22
# ╟─43d68541-84a5-4a63-9d8f-43783cc27ccc
# ╟─90a47e0b-b911-4728-80b5-6ed74607833d
# ╟─5d79b8c1-beea-4ff9-9830-0f5e1c4ef29f
# ╟─a5bfd185-aa77-4e46-a6b6-d43c4785a7fa
# ╟─a7b6ecbd-1407-44dd-809e-33311970af12
# ╟─b389a6a0-dc8e-4c6f-8a82-4f8878ffe879
# ╟─e0e97839-884a-49ed-bee4-f1f2ace5f5e0
# ╠═bcdd60b8-e0d8-4a70-88d6-725269447c9b
# ╟─9de99f4a-9970-4be1-9e16-e64ed4e10277
# ╟─518e7077-d61b-4f60-987f-d556e3eb1d0d
# ╠═1337513f-995f-4dfa-827d-797a5d2e5e1a
# ╟─f5e789b2-a62e-4818-90c3-76f39ea11aaa
# ╟─efa7736c-22c0-410e-94da-1df315f22bbf
# ╠═e9df3afb-fa04-440f-9664-3496da85696b
# ╟─58b7267d-491d-40f0-b4ba-27ed0c9cc855
# ╟─ac76b646-7c28-4384-9f04-5e4de5df154f
# ╠═83a14158-33d1-4f16-85e1-2726c8fbbdfc
# ╟─4b31dca2-0195-4899-8a3a-e9772fabf495
# ╟─79e0deab-1e36-4863-ad10-187ed8555c72
# ╟─66d385ba-9c6e-4378-b4e0-e54a4df346a5
# ╟─414cdd7a-fa93-4554-8b76-8f3637b08406
# ╟─6bf478d1-207b-4af5-b006-8a07002c8c4e
# ╟─db799fa2-0e48-43ee-9ee1-80ff8d2e5de7
# ╟─eb3a6009-e181-443c-bb77-021e867030e4
# ╠═521f5ffa-2c22-44c5-8bdb-67410431ca2e
# ╟─842bf89d-45eb-462d-ba74-ca260a8b177d
# ╟─f9b35e98-347f-4ebd-a690-790c7b0e03d8
# ╟─80fa8831-924f-4093-a89c-bf8fc440da6b
# ╟─4a3630ca-c8dd-4e81-8ee2-bb0fc6b01a93
# ╟─77353ccd-8788-4c68-9560-03f8c8100de1
# ╟─28542b8c-0b1c-498a-9b59-2a008c86c532
# ╠═a49c4e3a-f233-4d24-9ca3-2b513fe1def7
# ╠═c5971140-9caa-4685-be6d-c2c01b724460
# ╠═2438af36-f51d-44a7-a314-383e70b51271
# ╠═4548d672-97ce-4b44-894c-3d75f693ed59
# ╠═d11d201a-989d-4b0f-a9e3-3611ad54f283
# ╟─99705bd4-8f55-42b9-b3a8-3470ed153f04
# ╠═8dbfd2c8-139d-4c22-a35c-d48ee92472f8
# ╠═9000cf83-d1be-4c6c-b53e-f0f7e24eac60
# ╠═7570cc23-8536-4648-8904-07bc776b1c94
# ╟─fb49b357-a334-4415-8c08-c936745918ae
# ╠═7a233dd0-620f-4d4c-9501-a52891ea3de5
# ╠═0d04e925-aa17-4753-a4f6-797b0742ff1d
# ╠═89e01d48-bd57-40c1-88e2-aef7f2fa78c2
# ╟─2d08e21e-34ed-4c52-b076-67fc39d63f80
# ╠═7e4e3906-f79a-4ecf-84da-f84c44c5476a
# ╠═53c6e347-e4f3-4a75-9bf0-cb5cd3cdbe0b
# ╠═e519133a-78b3-47ad-b375-3a5ec453b229
# ╠═b9e13f67-36a6-4a4f-afa9-5e0321cf702b
# ╠═b1eebc9c-734b-4c2a-b08e-d4196b2f9cbd
# ╟─70f1127d-64ba-4824-895d-252e71c8d4de
