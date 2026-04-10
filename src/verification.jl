using LinearAlgebra, Printf

# Two-stage verification:
#   • verify_elem_forces  — pure vs. Ferrite ground truth, per element
#   • verify_poly_force   — global polynomial K1+K2+K3 vs. direct assembly
# Both abort/warn loudly on mismatch.

# Compare elem_force_pure against elem_force_ferrite on every element with
# random displacements. Aborts if relative error exceeds `tol`.
function verify_elem_forces(dh, cv, elems, mat;
                            ntests=3, scale=1e-2, tol=1e-10)
    println("=== Element-level verification ===")
    maxerr = 0.0

    for (eidx, cell) in enumerate(CellIterator(dh))
        geom = elems[eidx]
        nbf  = getnbasefunctions(cv)

        for t in 1:ntests
            s  = scale * (0.1^(t - 1))
            ue = s * randn(nbf)

            f_ref  = elem_force_ferrite(ue, cv, mat, cell)
            f_pure = elem_force_pure(ue, geom, mat)

            relerr = norm(f_pure - f_ref) / max(norm(f_ref), 1e-30)
            maxerr = max(maxerr, relerr)
        end
    end

    @printf("  Max element-level relative error: %.2e\n", maxerr)
    if maxerr > tol
        error("ABORT: pure element force doesn't match Ferrite (err = $maxerr)")
    end
    println("  ✓ Pure function matches Ferrite to machine precision")
    return maxerr
end

# Compare polynomial evaluation against direct element-by-element assembly
# at multiple displacement scales. Should be machine precision for SVK.
function verify_poly_force(elems, mat, K1, K2, K3, nDOF;
                           ntests=5, max_scale=1e-2)
    println("=== Global polynomial verification ===")

    for t in 1:ntests
        s = max_scale * (0.1^(t - 1))
        q = s * randn(nDOF)

        # Direct element-by-element assembly
        f_dir = zeros(nDOF)
        for geom in elems
            ue = q[geom.gdofs]
            fe = elem_force_pure(ue, geom, mat)
            for (a, gd) in enumerate(geom.gdofs)
                f_dir[gd] += fe[a]
            end
        end

        # Polynomial evaluation
        f_pol = eval_poly_force(K1, K2, K3, q)

        abserr = norm(f_pol - f_dir)
        relerr = abserr / max(norm(f_dir), 1e-30)
        ok     = relerr < 1e-8 ? "✓" : "✗"

        @printf("  %s |q|=%.2e : ‖Δf‖=%.2e  rel=%.2e\n",
                ok, norm(q), abserr, relerr)
    end
end
