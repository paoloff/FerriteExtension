using SparseArrays

# Loop over all elements, extract local AD tensors, scatter into global COO.
# Returns (K1, K2, K3) — K1 as SparseMatrixCSC, K2/K3 as SparseT3/SparseT4.
function assemble_poly_tensors(elems::Vector{ElementGeom}, mat::SVKMaterial,
                               nDOF::Int; tol=1e-15)
    Ir1 = Int[];     Ic1 = Int[];     V1  = Float64[]
    Ir2 = Int[];     Ic2 = Int[];     Is2 = Int[];     V2 = Float64[]
    Ir3 = Int[];     Ic3 = Int[];     Is3 = Int[];     It3 = Int[];     V3 = Float64[]

    nelem = length(elems)
    for (eidx, geom) in enumerate(elems)
        if eidx == 1 || eidx == nelem || eidx % max(1, nelem ÷ 5) == 0
            println("  Element $eidx / $nelem")
        end

        K1e, K2e, K3e = extract_elem_tensors(geom, mat)
        gd   = geom.gdofs
        ndof = length(gd)

        # Scatter K1
        for a in 1:ndof, b in 1:ndof
            v = K1e[a, b]
            abs(v) > tol || continue
            push!(Ir1, gd[a]); push!(Ic1, gd[b]); push!(V1, v)
        end

        # Scatter K2 — all (i,j,k) including off-diagonal
        for a in 1:ndof, b in 1:ndof, c in 1:ndof
            v = K2e[a, b, c]
            abs(v) > tol || continue
            push!(Ir2, gd[a]); push!(Ic2, gd[b]); push!(Is2, gd[c]); push!(V2, v)
        end

        # Scatter K3 — all (i,j,k,l) including off-diagonal
        for a in 1:ndof, b in 1:ndof, c in 1:ndof, d in 1:ndof
            v = K3e[a, b, c, d]
            abs(v) > tol || continue
            push!(Ir3, gd[a]); push!(Ic3, gd[b])
            push!(Is3, gd[c]); push!(It3, gd[d]); push!(V3, v)
        end
    end

    K1 = sparse(Ir1, Ic1, V1, nDOF, nDOF)
    K2 = SparseT3(Ir2, Ic2, Is2, V2, nDOF)
    K3 = SparseT4(Ir3, Ic3, Is3, It3, V3, nDOF)
    return K1, K2, K3
end

# Consistent mass matrix from pre-extracted shape values.
# Dimension-independent (uses geom.sdim).
function assemble_mass(elems::Vector{ElementGeom}, nDOF::Int, ρ::Float64)
    Ir = Int[]; Ic = Int[]; V = Float64[]

    for geom in elems
        dim = geom.sdim
        nbf = size(geom.N, 2)
        nqp = size(geom.N, 3)

        for qp in 1:nqp
            w = geom.dΩ[qp]
            for a in 1:nbf, b in 1:nbf
                d = 0.0
                for I in 1:dim
                    d += geom.N[I, a, qp] * geom.N[I, b, qp]
                end
                v = ρ * d * w
                if abs(v) > 1e-25
                    push!(Ir, geom.gdofs[a]); push!(Ic, geom.gdofs[b])
                    push!(V, v)
                end
            end
        end
    end

    return sparse(Ir, Ic, V, nDOF, nDOF)
end
