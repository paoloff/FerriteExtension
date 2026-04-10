using Ferrite, Tensors

# Element internal force — two implementations:
#   • elem_force_ferrite : uses Ferrite's CellValues + Tensors.jl directly
#                          (ground truth for verification, any dimension)
#   • elem_force_pure    : pure-functional, ForwardDiff-compatible to any order
#                          (reads ElementGeom; works in 1D, 2D, 3D via sdim)

# Reference implementation. Uses Ferrite/Tensors operations — not AD-friendly,
# but mathematically transparent. Used by verify_element_forces as ground truth.
function elem_force_ferrite(ue::Vector{Float64}, cv::CellValues,
                            mat::SVKMaterial, cell)
    reinit!(cv, cell)
    nbf = getnbasefunctions(cv)
    fe  = zeros(nbf)

    for qp in 1:getnquadpoints(cv)
        dΩ  = getdetJdV(cv, qp)
        ∇u  = function_gradient(cv, qp, ue)
        F   = one(∇u) + ∇u
        FᵀF = tdot(F)
        E   = (FᵀF - one(FᵀF)) / 2
        S   = mat.λ * tr(E) * one(E) + 2 * mat.μ * E
        P   = F ⋅ S

        for a in 1:nbf
            fe[a] += (P ⊡ shape_gradient(cv, qp, a)) * dΩ
        end
    end
    return fe
end


# Pure-functional element internal force.
# Only arithmetic on plain arrays — Dual numbers flow through cleanly.
# Reads `geom.sdim` and uses generic dim×dim loops, so it works in any dim.
function elem_force_pure(ue::AbstractVector{T}, geom::ElementGeom,
                         mat::SVKMaterial) where T
    dim  = geom.sdim
    nbf  = size(geom.∇N, 2)
    nqp  = size(geom.∇N, 3)
    ndof = length(ue)

    fe = zeros(T, ndof)

    @inbounds for qp in 1:nqp
        w = geom.dΩ[qp]

        # ∇u = Σ_a ue[a] * ∇N_a
        ∇u = zeros(T, dim, dim)
        for a in 1:nbf
            ua = ue[a]
            idx = 1
            for I in 1:dim, J in 1:dim
                ∇u[I, J] += ua * geom.∇N[idx, a, qp]
                idx += 1
            end
        end

        # F = I + ∇u
        F = zeros(T, dim, dim)
        for I in 1:dim, J in 1:dim
            F[I, J] = ∇u[I, J] + (I == J ? one(T) : zero(T))
        end

        # FᵀF
        FᵀF = zeros(T, dim, dim)
        for I in 1:dim, J in 1:dim, K in 1:dim
            FᵀF[I, J] += F[K, I] * F[K, J]
        end

        # E = ½(FᵀF − I)
        E = zeros(T, dim, dim)
        for I in 1:dim, J in 1:dim
            E[I, J] = (FᵀF[I, J] - (I == J ? one(T) : zero(T))) / 2
        end

        # S = λ tr(E) I + 2μ E
        trE = zero(T)
        for I in 1:dim
            trE += E[I, I]
        end
        S = zeros(T, dim, dim)
        for I in 1:dim, J in 1:dim
            S[I, J] = 2 * mat.μ * E[I, J] + (I == J ? mat.λ * trE : zero(T))
        end

        # P = F · S
        P = zeros(T, dim, dim)
        for I in 1:dim, J in 1:dim, K in 1:dim
            P[I, J] += F[I, K] * S[K, J]
        end

        # fe[a] += (P : ∇N_a) * w
        for a in 1:nbf
            c = zero(T)
            idx = 1
            for I in 1:dim, J in 1:dim
                c += P[I, J] * geom.∇N[idx, a, qp]
                idx += 1
            end
            fe[a] += c * w
        end
    end

    return fe
end
