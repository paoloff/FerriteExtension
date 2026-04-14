# Analytical SVK polynomial tensor extraction — no AD.

# K2raw and K3raw below are the un-symmetrized coefficients of u_β u_γ and
# u_β u_p u_q; K2e and K3e are obtained by symmetrizing over the trailing
# indices so they match the ½-Hessian / (1/6)-third-derivative convention
# of the AD version entry-by-entry.

function extract_elem_tensors_svk(geom::ElementGeom, mat::SVKMaterial)
    dim  = geom.sdim
    nqp  = size(geom.∇N, 3)
    ndof = length(geom.gdofs)
    λ = mat.λ;  μ = mat.μ

    K1e   = zeros(ndof, ndof)
    K2raw = zeros(ndof, ndof, ndof)
    K3raw = zeros(ndof, ndof, ndof, ndof)

    # per-qp scratch
    B   = zeros(dim, dim, ndof)          # B[I,J,a] = ∂N_a/∂X component (I,J)
    C   = zeros(dim, dim, ndof)          # C^a
    Q   = zeros(dim, dim, ndof, ndof)    # Q[M,J,a,β] = Σ_I B[I,J,a] B[I,M,β]
    Gpq = zeros(dim, dim)
    Dpq = zeros(dim, dim)

    @inbounds for qp in 1:nqp
        w = geom.dΩ[qp]

        # --- unpack B^a_{IJ} from flat storage --------------------------
        for a in 1:ndof
            idx = 1
            for I in 1:dim, J in 1:dim
                B[I, J, a] = geom.∇N[idx, a, qp]
                idx += 1
            end
        end

        # --- C^a = λ·tr(B^a)·I + 2μ·sym(B^a) ---------------------------
        for a in 1:ndof
            trB = 0.0
            for I in 1:dim
                trB += B[I, I, a]
            end
            for J in 1:dim, I in 1:dim
                sym_IJ = 0.5 * (B[I, J, a] + B[J, I, a])
                C[I, J, a] = 2μ * sym_IJ + (I == J ? λ * trB : 0.0)
            end
        end

        # --- K1e[a,β] += w · Σ_{IJ} B^a_{IJ} C^β_{IJ} ------------------
        for β in 1:ndof, a in 1:ndof
            s = 0.0
            for J in 1:dim, I in 1:dim
                s += B[I, J, a] * C[I, J, β]
            end
            K1e[a, β] += w * s
        end

        # --- K2 Part B (from ∇u·S1): Σ_{IMJ} B^a_{IJ} B^β_{IM} C^γ_{MJ} -
        for γ in 1:ndof, β in 1:ndof, a in 1:ndof
            s = 0.0
            for J in 1:dim, M in 1:dim, I in 1:dim
                s += B[I, J, a] * B[I, M, β] * C[M, J, γ]
            end
            K2raw[a, β, γ] += w * s
        end

        # --- Q[M,J,a,β] = Σ_I B[I,J,a] B[I,M,β] ------------------------
        for β in 1:ndof, a in 1:ndof
            for J in 1:dim, M in 1:dim
                s = 0.0
                for I in 1:dim
                    s += B[I, J, a] * B[I, M, β]
                end
                Q[M, J, a, β] = s
            end
        end

        # --- (p,q) sweep: K2 Part A and K3 both need D^{pq} ------------
        for qidx in 1:ndof, p in 1:ndof
            # G^{pq}[I,J] = ½ Σ_K B[K,I,p] B[K,J,qidx]
            trG = 0.0
            for J in 1:dim, I in 1:dim
                s = 0.0
                for K in 1:dim
                    s += B[K, I, p] * B[K, J, qidx]
                end
                Gpq[I, J] = 0.5 * s
            end
            for I in 1:dim
                trG += Gpq[I, I]
            end
            for J in 1:dim, I in 1:dim
                Dpq[I, J] = 2μ * Gpq[I, J] + (I == J ? λ * trG : 0.0)
            end

            # K2 Part A: K2raw[a,p,qidx] += w · Σ_{IJ} B^a_{IJ} Dpq[I,J]
            for a in 1:ndof
                s = 0.0
                for J in 1:dim, I in 1:dim
                    s += B[I, J, a] * Dpq[I, J]
                end
                K2raw[a, p, qidx] += w * s
            end

            # K3: K3raw[a,β,p,qidx] += w · Σ_{MJ} Q[M,J,a,β] · Dpq[M,J]
            #     (equivalent to Σ_{IMJ} B^a_{IJ} B^β_{IM} Dpq[M,J])
            for β in 1:ndof, a in 1:ndof
                s = 0.0
                for J in 1:dim, M in 1:dim
                    s += Q[M, J, a, β] * Dpq[M, J]
                end
                K3raw[a, β, p, qidx] += w * s
            end
        end
    end

    # --- symmetrize K2 in (β,γ) to match the ½·Hessian convention ------
    K2e = similar(K2raw)
    @inbounds for γ in 1:ndof, β in 1:ndof, a in 1:ndof
        K2e[a, β, γ] = 0.5 * (K2raw[a, β, γ] + K2raw[a, γ, β])
    end

    # --- symmetrize K3 in (β,γ,δ) to match the (1/6)·third-deriv convention
    K3e = similar(K3raw)
    @inbounds for δ in 1:ndof, γ in 1:ndof, β in 1:ndof, a in 1:ndof
        K3e[a, β, γ, δ] = (K3raw[a, β, γ, δ] + K3raw[a, β, δ, γ] +
                           K3raw[a, γ, β, δ] + K3raw[a, γ, δ, β] +
                           K3raw[a, δ, β, γ] + K3raw[a, δ, γ, β]) / 6
    end

    return K1e, K2e, K3e
end
