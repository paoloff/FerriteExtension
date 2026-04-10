using SparseArrays

# Sparse rank-3 / rank-4 tensors in COO format.
# Used for the quadratic (K2) and cubic (K3) coefficients of the Taylor
# expansion of the internal force around q=0.

# T[I[k], J[k], K[k]] = V[k]
struct SparseT3
    I::Vector{Int}
    J::Vector{Int}
    K::Vector{Int}
    V::Vector{Float64}
    N::Int   # tensor side length (N×N×N)
end

# T[I[k], J[k], K[k], L[k]] = V[k]
struct SparseT4
    I::Vector{Int}
    J::Vector{Int}
    K::Vector{Int}
    L::Vector{Int}
    V::Vector{Float64}
    N::Int   # tensor side length (N×N×N×N)
end

# Evaluate the polynomial internal force
#   f(q) = K1·q + K2(q⊗q) + K3(q⊗q⊗q)
function eval_poly_force(K1::SparseMatrixCSC, K2::SparseT3, K3::SparseT4,
                         q::AbstractVector)
    f = K1 * q
    @inbounds for k in eachindex(K2.V)
        f[K2.I[k]] += K2.V[k] * q[K2.J[k]] * q[K2.K[k]]
    end
    @inbounds for k in eachindex(K3.V)
        f[K3.I[k]] += K3.V[k] * q[K3.J[k]] * q[K3.K[k]] * q[K3.L[k]]
    end
    return f
end

# Drop entries that touch any constrained DOF.
function filter_cdofs(T::SparseT3, cdofs::Set{Int})
    m = [!(T.I[k] in cdofs || T.J[k] in cdofs || T.K[k] in cdofs)
         for k in eachindex(T.V)]
    return SparseT3(T.I[m], T.J[m], T.K[m], T.V[m], T.N)
end

function filter_cdofs(T::SparseT4, cdofs::Set{Int})
    m = [!(T.I[k] in cdofs || T.J[k] in cdofs ||
           T.K[k] in cdofs || T.L[k] in cdofs)
         for k in eachindex(T.V)]
    return SparseT4(T.I[m], T.J[m], T.K[m], T.L[m], T.V[m], T.N)
end
