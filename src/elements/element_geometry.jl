using Ferrite

# ElementGeom — pre-extracted geometry for a single element.
# Stored as plain Float64 arrays so ForwardDiff can trace through it without
# touching any Ferrite type. One ElementGeom per cell in the mesh.
# Fields:
#   gdofs : local→global DOF map (length ndof_e)
#   ∇N    : (dim*dim, nbf, nqp) flattened shape-function gradients
#   N     : (dim, nbf, nqp) shape-function values
#   dΩ    : (nqp,) det(J) * quadrature weight at each qp
#   sdim  : spatial dimension (1, 2, or 3)

struct ElementGeom
    gdofs::Vector{Int}
    ∇N::Array{Float64, 3}
    N::Array{Float64, 3}
    dΩ::Vector{Float64}
    sdim::Int
end

# Loop over all cells, extract gradients/values/weights from Ferrite's CellValues
# into plain arrays. Returns one ElementGeom per cell.
function extract_element_geoms(dh, cv)
    elems = ElementGeom[]

    for cell in CellIterator(dh)
        reinit!(cv, cell)
        nqp = getnquadpoints(cv)
        nbf = getnbasefunctions(cv)
        gdofs = copy(celldofs(cell))

        # spatial dim from a sample shape gradient (Tensor{2, dim})
        dim = size(shape_gradient(cv, 1, 1), 1)

        ∇N = zeros(dim * dim, nbf, nqp)
        N  = zeros(dim, nbf, nqp)
        dΩ = zeros(nqp)

        for qp in 1:nqp
            dΩ[qp] = getdetJdV(cv, qp)
            for a in 1:nbf
                # gradient: Tensor{2, dim} → flatten (dim*dim,)
                g = shape_gradient(cv, qp, a)
                idx = 1
                for I in 1:dim, J in 1:dim
                    ∇N[idx, a, qp] = g[I, J]
                    idx += 1
                end
                # value: Vec{dim}
                v = shape_value(cv, qp, a)
                for I in 1:dim
                    N[I, a, qp] = v[I]
                end
            end
        end

        push!(elems, ElementGeom(gdofs, ∇N, N, dΩ, dim))
    end

    return elems
end
