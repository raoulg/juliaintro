### A Pluto.jl notebook ###
# v0.19.9

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

# ╔═╡ 519d346c-0351-11ed-1af6-9705121dd49c
using Parquet, DataFrames, PlutoUI

# ╔═╡ 877eb5b3-50af-4ea6-96b2-0c7cb99ac36c
md"First, we will have a look at DataFrames. Typically, you will start loading in data and exploring the data"

# ╔═╡ eda1134a-cd1a-49b1-a5e9-8a2e33fb6baf
path = "data/palmerpenguins.parq"

# ╔═╡ b3c924c9-eb99-46df-9fcd-aa293b607155
penguin_url = "https://education.rstudio.com/blog/2020/07/palmerpenguins-cran/gorman-penguins.jpg"

# ╔═╡ 7176efbf-a0b3-4023-a279-638199693530
md"So, I have a file with data. It is the `palmerpenguins.parq` file inside the data folder. The Palmer Penguins are the new Iris dataset. Some [people](https://education.rstudio.com/blog/2020/07/palmerpenguins-cran/)  measured all kinds of features on three types of penguins. Besides that the penguins are cute, the dataset has a lot of interesting statistical properties that make this a very nice dataset to explore some basic machine learning. $(Resource(penguin_url))"

# ╔═╡ 6e1a8902-c876-4cee-859a-7fec347765b4
raw_data = DataFrame(read_parquet(path))

# ╔═╡ df69d40a-6c67-4f5a-a0a7-8f0b9f7e81e9
md"This is how you load a file. You could also use a package like CVS (with `using CSV` if you insist on using a fileformat that is about 10-50 times slower than parquet). If you hover your mouse just below the column names, you will see the (guessed) datatypes of the column (eg string or float64)"

# ╔═╡ f0a9646f-8fb1-42f6-8dca-0f27819f9199
describe(raw_data)

# ╔═╡ 1eaa7859-f70a-4fe1-8911-56f26ee31965
md"""As you can notice, in some columns there is missing data. Let's zoom in on that a little bit. You can either scroll in the output of `describe` above, or specify the `:nmissing` column. Note the syntax with the `:` , like `:Species`. 

This is Julia syntax for "This is a Symbol, not a string"
"""

# ╔═╡ ba2ac3f0-c726-46be-be31-668f6574d586
describe(raw_data, :nmissing)

# ╔═╡ d89bd760-5d54-4b6d-910e-32ca501eb6f0
md"""
Ok, so, the missing data is not too bad. It is mainly the `:Comments` that are missing. Lets drop them for now. After dropping that column, we will drop all missing columns. We will wrap these two commands in a `begin-end` block, because Pluto's reactive interface likes to parse every cell as a whole.
"""

# ╔═╡ 54e74742-9a38-4362-accf-f9f39b9cd8e4
begin
	data = select(raw_data, Not(:Comments))
	dropmissing!(data, disallowmissing=true)
end

# ╔═╡ bce3ab0a-75e5-4fc9-87e1-f858f30f2cc9
md"""
You might be curious what the `disallowmissing` means. This is a great moment to introduce you to the concept of the LiveDocs. Just click on the `Live Docs` right below in the notebook, and place your cursor inside the `disallowmissing` command, or in the `dropmissing`. Nice, isnt it?

Also note how I used the `dropmissing!` instead of `df = dropmissing(data)`. The `!` at the end of a function typically means the same as the `inplace=True` in Python: it modifies the object passed to it directly.
"""

# ╔═╡ df3b67a0-3ae0-45bf-bedc-ddf28f200b7c
names(raw_data)

# ╔═╡ 6e8de152-f3db-45a8-b503-dddd51a0e21a
rename!(data, [1 => :study, 
	2=> :sample, 
	3 => :species, 
	4 => :region,
	5 => :island, 
	6 => :stage, 
	7 => :id, 
	8 => :clutch, 
	9 => :date, 
	10 => :cullen,
	11 => :culdep, 
	12 => :fliplen, 
	13 => :mass, 
	14 => :sex, 
	15 => :Δ15, 
	16 => :Δ13])

# ╔═╡ add6dba3-cfff-45a3-905e-aca1899f9f8b
md"""next thing, I dont like the long names as columns. How wants to type out 
"Delta 13 C (o/oo)" when referring to a column? 

You see a few things happing that might be new to you.

First thing, the `=>`. Think of it as a mapping. In Python, you would use a dictionary for this, eg. {1 : "study"}. In Julia, it's called a Pair (use the LiveDocs if you want to know more).

Next, you see me using `rename!` (so you already know what that does) and passing it both `data` and a list.

In Python, lists are very slow. Like, a factor 1000 slower than a numpy array. In Julia, the datatype of objects created with `[` and `]` are `Array`.

As you can see in the cell below, I used a vector with pairs of `Int64` and `Symbols`. The nice thing is, that we can discern this from, let's say, a Vector with `Pairs` with other types (eg `Pair{String, Float64}`).

Finally, note how I use the $\Delta$ symbol. You can use all sorts of symbols in Julia; just type \Delta (or \sigma etc) and press tab.
"""

# ╔═╡ ae3a06ea-0655-409c-9ba2-4509133f71b2
typeof([1=> :a,2 => :b])

# ╔═╡ fe4b37bd-fba2-4125-a6e1-94a954e708bd
md"""
Another thing I would like to change, is to strip the :species from the clutter. That way I can use it as a label when plotting. 

To do that, I will use the `split` function.

Now, what I'm going to do will look new to you if you are coming from Python. I will walk you though it, but first let me show it.
"""

# ╔═╡ 573bcc70-c7c2-4c54-86c7-3dcbbfac4092
function split_first(x::String)::String
	return string(split(x, " ")[1])
end

# ╔═╡ 2ec4854a-8d14-453f-882c-697b8c547e4c
transform!(data, :species => ByRow(split_first) => :species)

# ╔═╡ ef238bcd-e98d-4ec8-961c-56852db5f229
md"""
Ok, you already now that `transform!` will work inplace.
`transform` helps us mutate the dataframe.

There are multiple ways I could transfrom the `data`.
The second argument takes the form `cols => function => target_cols`, where we could ommit the `target_cols` if we wanted to.

The `cols` can either be a Symbol (like we do, with :species) or a Vector of column names. The function is split(x, " ")[1]. Because that would give us a type `SubString`, and Parquet does not know how to write that, we need to cast to `String`.

So my function (note the syntax, it is pretty close to the python syntax) takes in a `x` and outputs the first item of the split (Julia has 1-based indexed Arrays, where Python has 0-based indexing.)

Finally, I wrap this in a ByRow. For performance reasons, you could swap this with a `map` function (see livedocs of ByRow) but for now, I think `ByRow` is more readable.

Note that my `target_col` is identical to `col`, because I want to overwrite the original column.
"""

# ╔═╡ c7a4141c-8418-4922-91e5-e1125f0f8987
@bind write_to_file CheckBox(default=false)

# ╔═╡ c2d49fca-325b-4988-8da4-ae5bdc20b283
write_to_file ? write_parquet("data/processed.parq", data) : "not writing"

# ╔═╡ 1e48b64d-685a-46a6-b148-8da53d201b74
md"""
I'm done preprocessing, and want to write my file.

I could have used an `if .... end` block, but I just wanted to show the `... ? ... : ...` syntax, which translates `a ? b : c` into `if a, evaluate b otherwise evaluate c`
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Parquet = "626c502c-15b0-58ad-a749-f091afb673ae"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
DataFrames = "~1.3.4"
Parquet = "~0.8.4"
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5f5a975d996026a8dd877c35fe26a7b8179c02ba"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.CodecZstd]]
deps = ["CEnum", "TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "849470b337d0fa8449c21061de922386f32949d9"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.7.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

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

[[deps.Decimals]]
git-tree-sha1 = "e98abef36d02a0ec385d68cd7dadbce9b28cbd88"
uuid = "abce61dc-4473-55a0-ba07-351d65e31d42"
version = "0.4.1"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

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

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LittleEndianBase128]]
deps = ["Test"]
git-tree-sha1 = "2cad132b52c86e0ccfc75116362ab57f0047893a"
uuid = "1724a1d5-ab78-548d-94b3-135c294f96cf"
version = "0.3.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parquet]]
deps = ["CategoricalArrays", "CodecZlib", "CodecZstd", "DataAPI", "Dates", "Decimals", "LittleEndianBase128", "Missings", "Mmap", "SentinelArrays", "Snappy", "Tables", "Thrift"]
git-tree-sha1 = "2b718c2ad5c1df9880e36d99c28c33ada37d6eb4"
uuid = "626c502c-15b0-58ad-a749-f091afb673ae"
version = "0.8.4"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Snappy]]
deps = ["CEnum", "snappy_jll"]
git-tree-sha1 = "72bae53c0691f4b6fd259587dab8821ae0e025f6"
uuid = "59d4ed8c-697a-5b28-a4c7-fe95c22820f9"
version = "0.4.2"

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

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Thrift]]
deps = ["CodecZlib", "CodecZstd", "Distributed", "Sockets", "ThriftJuliaCompiler_jll", "TranscodingStreams"]
git-tree-sha1 = "3da5858caabd351ee845bb1d547e4d70dbc65b20"
uuid = "8d9c9c80-f77e-5080-9541-c6f69d204e22"
version = "0.8.4"

[[deps.ThriftJuliaCompiler_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "949a51ca85d31b063531eed49e38a6c9b9bae58b"
uuid = "815b9798-8dd0-5549-95cc-3cf7d01bce66"
version = "0.12.1+0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.snappy_jll]]
deps = ["Artifacts", "JLLWrappers", "LZO_jll", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "985c1da710b0e43f7c52f037441021dfd0e3be14"
uuid = "fe1e1685-f7be-5f59-ac9f-4ca204017dfd"
version = "1.1.9+1"
"""

# ╔═╡ Cell order:
# ╠═519d346c-0351-11ed-1af6-9705121dd49c
# ╟─877eb5b3-50af-4ea6-96b2-0c7cb99ac36c
# ╠═eda1134a-cd1a-49b1-a5e9-8a2e33fb6baf
# ╟─7176efbf-a0b3-4023-a279-638199693530
# ╟─b3c924c9-eb99-46df-9fcd-aa293b607155
# ╠═6e1a8902-c876-4cee-859a-7fec347765b4
# ╟─df69d40a-6c67-4f5a-a0a7-8f0b9f7e81e9
# ╠═f0a9646f-8fb1-42f6-8dca-0f27819f9199
# ╟─1eaa7859-f70a-4fe1-8911-56f26ee31965
# ╠═ba2ac3f0-c726-46be-be31-668f6574d586
# ╟─d89bd760-5d54-4b6d-910e-32ca501eb6f0
# ╠═54e74742-9a38-4362-accf-f9f39b9cd8e4
# ╟─bce3ab0a-75e5-4fc9-87e1-f858f30f2cc9
# ╠═df3b67a0-3ae0-45bf-bedc-ddf28f200b7c
# ╠═6e8de152-f3db-45a8-b503-dddd51a0e21a
# ╟─add6dba3-cfff-45a3-905e-aca1899f9f8b
# ╠═ae3a06ea-0655-409c-9ba2-4509133f71b2
# ╟─fe4b37bd-fba2-4125-a6e1-94a954e708bd
# ╠═2ec4854a-8d14-453f-882c-697b8c547e4c
# ╠═573bcc70-c7c2-4c54-86c7-3dcbbfac4092
# ╟─ef238bcd-e98d-4ec8-961c-56852db5f229
# ╠═c7a4141c-8418-4922-91e5-e1125f0f8987
# ╠═c2d49fca-325b-4988-8da4-ae5bdc20b283
# ╟─1e48b64d-685a-46a6-b148-8da53d201b74
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
