using Ferrite
using Tensors
using FerriteGmsh
using ForwardDiff
using SparseArrays
using LinearAlgebra
using Printf
using JSON

include("./src/material.jl")
include("./src/elements/element_geometry.jl")
include("./src/elements/element_force.jl")
include("./src/tensors/sparse_tensors.jl")
include("./src/tensors/extract_tensors.jl")
include("./src/tensors/assemble_tensors.jl")
include("./src/boundary_conditions.jl")
include("./src/tensors/export_tensors.jl")
include("./src/verification.jl")
include("./src/pmap_connector.jl")
