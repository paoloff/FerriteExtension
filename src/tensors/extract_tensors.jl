using ForwardDiff

# Polynomial tensor extraction via nested ForwardDiff.
# For a single element with displacements u ∈ R^ndof_e:
#   K1e = ∂f/∂u                       (Jacobian)
#   K2e = ½ ∂²f_i/∂u∂u                (Hessian per output / 2)
#   K3e = ⅙ ∂³f_i/∂u∂u∂u              (Jacobian of vec(Hessian) / 6)
# all evaluated at u = 0. For SVK these are exact (f is cubic in u).

# Extract the three local tensors for one element via nested ForwardDiff.
function extract_elem_tensors(geom::ElementGeom, mat::SVKMaterial)
    ndof = length(geom.gdofs)
    u0   = zeros(ndof)
    f(u) = elem_force_pure(u, geom, mat)

    # K1e: linear stiffness
    K1e = ForwardDiff.jacobian(f, u0)

    # K2e: ½ Hessian of each output component
    K2e = zeros(ndof, ndof, ndof)
    for i in 1:ndof
        Hi = ForwardDiff.hessian(u -> elem_force_pure(u, geom, mat)[i], u0)
        for j in 1:ndof, k in 1:ndof
            K2e[i, j, k] = Hi[j, k] / 2
        end
    end

    # K3e: ⅙ Jacobian of vec(Hessian)
    K3e = zeros(ndof, ndof, ndof, ndof)
    for i in 1:ndof
        vecHi(u) = vec(ForwardDiff.hessian(
            v -> elem_force_pure(v, geom, mat)[i], u))
        JH = ForwardDiff.jacobian(vecHi, u0)
        for l in 1:ndof, k in 1:ndof, j in 1:ndof
            K3e[i, j, k, l] = JH[j + (k - 1) * ndof, l] / 6
        end
    end

    return K1e, K2e, K3e
end
