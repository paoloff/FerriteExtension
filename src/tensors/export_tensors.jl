using SparseArrays, JSON, Printf

# Export all tensors as COO-format CSVs plus a metadata.json.
# Format is language-agnostic — can be loaded from Python, MATLAB, Julia, etc.

function export_tensors(dir::String, M::SparseMatrixCSC, K1::SparseMatrixCSC,
                        K2::SparseT3, K3::SparseT4; meta::Dict = Dict())
    mkpath(dir)

    # metadata
    if !isempty(meta)
        open(joinpath(dir, "metadata.json"), "w") do io
            JSON.print(io, meta, 2)
        end
    end

    # 2D sparse matrices (M, K1)
    for (fname, mat) in [("mass_coo.csv", M), ("stiffness_coo.csv", K1)]
        Ir, Ic, V = findnz(mat)
        open(joinpath(dir, fname), "w") do io
            println(io, "row,col,val")
            for k in eachindex(V)
                println(io, "$(Ir[k]),$(Ic[k]),$(V[k])")
            end
        end
    end

    # K2 (rank-3)
    open(joinpath(dir, "K2_coo.csv"), "w") do io
        println(io, "i,j,k,val")
        for k in eachindex(K2.V)
            println(io, "$(K2.I[k]),$(K2.J[k]),$(K2.K[k]),$(K2.V[k])")
        end
    end

    # K3 (rank-4)
    open(joinpath(dir, "K3_coo.csv"), "w") do io
        println(io, "i,j,k,l,val")
        for k in eachindex(K3.V)
            println(io, "$(K3.I[k]),$(K3.J[k]),$(K3.K[k]),$(K3.L[k]),$(K3.V[k])")
        end
    end

    println("Exported to: $dir/")
    for fname in readdir(dir)
        sz = filesize(joinpath(dir, fname))
        @printf("  %-25s %8d bytes\n", fname, sz)
    end
end
