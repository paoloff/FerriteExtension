using Ferrite, SparseArrays

# Dirichlet boundary conditions.
# Applied AFTER polynomial verification, on the unconstrained system.

# Zero out rows/columns of constrained DOFs in K1 and M, place 1 on diagonal.
# Returns the list of constrained DOFs for later filtering of K2/K3.
function apply_dirichlet!(K1::SparseMatrixCSC, M::SparseMatrixCSC, ch)
    cdofs = Ferrite.prescribed_dofs(ch)
    println("  Constrained DOFs: $(length(cdofs))")

    for d in cdofs
        K1[d, :] .= 0; K1[:, d] .= 0; K1[d, d] = 1.0
        M[d, :]  .= 0; M[:, d]  .= 0; M[d, d]  = 1.0
    end

    return cdofs
end

# Drop entries from K2/K3 that touch any constrained DOF.
function apply_dirichlet(K2::SparseT3, K3::SparseT4, cdofs)
    s = Set(cdofs)
    return filter_cdofs(K2, s), filter_cdofs(K3, s)
end
