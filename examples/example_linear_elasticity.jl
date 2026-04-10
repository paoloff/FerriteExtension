#=
    Compares the polynomial tensor formulation (K1 from SVK linearized at
    u=0 via ForwardDiff) against Ferrite's standard linear-elasticity
    assembly (C ⊡ ∇ˢʸᵐN). They must match to machine precision.
 =#

using Downloads: download

include("../includes.jl")


######################################
######################################
######################################

# PART A: Standard Ferrite linear elasticity
# Shared parameters 
E = 1.0 #200.0e3   # MPa
ν = 0.3
ρ = 1.0 #7.85e-9

# Mesh: Ferrite logo (Triangle mesh from .geo file) 
println("\nDownloading and loading Ferrite logo mesh...")
logo  = "logo.geo"
asset = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
isfile(logo) || download(string(asset, logo), logo)

grid = togrid(logo)

addfacetset!(grid, "top",    x -> x[2] ≈ 1.0)
addfacetset!(grid, "left",   x -> abs(x[1]) < 1.0e-6)
addfacetset!(grid, "bottom", x -> abs(x[2]) < 1.0e-6)

# Interpolation + quadrature 
sdim  = 2
order = 1
ip    = Lagrange{RefTriangle, order}()^sdim
qr    = QuadratureRule{RefTriangle}(1)
fqr   = FacetQuadratureRule{RefTriangle}(1)
cv    = CellValues(qr, ip)
fv    = FacetValues(fqr, ip)

# DOF handler (shared) 
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

nDOF = ndofs(dh)
@info "Shared mesh" nDOF nelems=getncells(grid) element="Triangle (Ferrite logo)"

# BCs: bottom fixes uy, left fixes ux 
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> 0.0, 1))
close!(ch)

# Standard elasticity tensor C 
μ = E / (2(1 + ν))
K = E / (3(1 - 2 * ν))

C = gradient(ϵ -> 2 * μ * dev(ϵ) + 3 * K * vol(ϵ),
             zero(SymmetricTensor{2, 2}))

# Standard stiffness assembly 
function assemble_cell_std!(Ke, cv, C)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        for i in 1:getnbasefunctions(cv)
            ∇Nᵢ = shape_gradient(cv, qp, i)
            for j in 1:getnbasefunctions(cv)
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cv, qp, j)
                Ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return Ke
end

function assemble_global_std!(K, dh, cv, C)
    nbf = getnbasefunctions(cv)
    Ke  = zeros(nbf, nbf)
    asm = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0.0)
        assemble_cell_std!(Ke, cv, C)
        assemble!(asm, celldofs(cell), Ke)
    end
    return K
end

println("Assembling standard stiffness matrix K_std...")
K_std = allocate_matrix(dh)
assemble_global_std!(K_std, dh, cv, C)
@printf("  K_std: %d×%d, nnz=%d\n", size(K_std)..., nnz(K_std))

# External force: traction on top edge 
function assemble_traction!(f, dh, fset, fv, t_fn)
    fe = zeros(getnbasefunctions(fv))
    for facet in FacetIterator(dh, fset)
        reinit!(fv, facet)
        fill!(fe, 0.0)
        xc = getcoordinates(facet)
        for qp in 1:getnquadpoints(fv)
            x  = spatial_coordinate(fv, qp, xc)
            t  = t_fn(x)
            dΓ = getdetJdV(fv, qp)
            for i in 1:getnbasefunctions(fv)
                Nᵢ = shape_value(fv, qp, i)
                fe[i] += t ⋅ Nᵢ * dΓ
            end
        end
        assemble!(f, celldofs(facet), fe)
    end
    return f
end

# Small traction → keeps strains in the linear regime where
# the SVK geometric-nonlinearity contribution is negligible.
traction(x) = Vec(0.0, 20.0 * x[1])

f_ext = zeros(nDOF)
assemble_traction!(f_ext, dh, getfacetset(grid, "top"), fv, traction)
@printf("  ‖f_ext‖ = %.4e\n", norm(f_ext))

# Solve standard problem 
println("Solving standard system...")
K_std_bc = copy(K_std)
f_std_bc = copy(f_ext)
apply!(K_std_bc, f_std_bc, ch)
u_std = K_std_bc \ f_std_bc
@printf("  ‖u_std‖    = %.6e\n", norm(u_std))
@printf("  max|u_std| = %.6e\n", maximum(abs.(u_std)))


######################################
######################################
######################################

# PART B: Polynomial tensor formulation (ForwardDiff AD)

# Match plane-strain / 3D-Lamé form: Tensors.jl `dev`/`vol` use tr/3, so
# σ = 2μ ε + λ_3D tr(ε) I — i.e. plane_stress=false.
mat = SVKMaterial(; E=E, ν=ν, ρ=ρ, plane_stress=false)

println("\nExtracting element geometry...")
elems = extract_element_geoms(dh, cv)
println("  $(length(elems)) elements, sdim=$(elems[1].sdim)")

verify_elem_forces(dh, cv, elems, mat)

println("\n=== Extracting K1, K2, K3 via ForwardDiff ===")
K1, K2, K3 = assemble_poly_tensors(elems, mat, nDOF);
@printf("  K1: nnz=%d\n", nnz(K1))
@printf("  K2: %d entries\n", length(K2.V))
@printf("  K3: %d entries\n", length(K3.V))

println()
verify_poly_force(elems, mat, K1, K2, K3, nDOF)

# Solve polynomial system (K1 * u = f) 
println("\nSolving polynomial system (K1 * u = f)...")
K1_bc    = copy(K1)
f_pol_bc = copy(f_ext)
apply!(K1_bc, f_pol_bc, ch)
u_pol = K1_bc \ f_pol_bc
@printf("  ‖u_pol‖    = %.6e\n", norm(u_pol))
@printf("  max|u_pol| = %.6e\n", maximum(abs.(u_pol)))


######################################
######################################
######################################

# PART C: Comparison
println("\n--- Stiffness matrix comparison ---")
ΔK     = K1 - K_std
abserr = norm(ΔK, Inf)
relerr = abserr / norm(K_std, Inf)
@printf("  ‖K1 - K_std‖_∞ = %.4e\n", abserr)
@printf("  Relative error  = %.4e\n", relerr)
println(relerr < 1e-10 ?
        "  ✓ Stiffness matrices match to machine precision" :
        "  ✗ Stiffness matrices differ — check material parameters")

println("\n--- Displacement comparison ---")
abserr = norm(u_pol - u_std)
relerr = abserr / norm(u_std)
@printf("  ‖u_pol - u_std‖ = %.4e\n", abserr)
@printf("  Relative error   = %.4e\n", relerr)
println(relerr < 1e-10 ?
        "  ✓ Displacement solutions match to machine precision" :
        "  ✗ Displacement solutions differ")

# Magnitudes of K2/K3 (sanity check for the linear regime) 
println("\n--- Nonlinear tensor magnitudes ---")
if length(K2.V) > 0
    @printf("  max|K2 entries| = %.4e\n", maximum(abs.(K2.V)))
else
    println("  K2 is empty (no quadratic terms)")
end
if length(K3.V) > 0
    @printf("  max|K3 entries| = %.4e\n", maximum(abs.(K3.V)))
else
    println("  K3 is empty (no cubic terms)")
end

# Polynomial force at the linear solution should match K1*u
f_lin     = K1 * u_pol
f_full    = eval_poly_force(K1, K2, K3, u_pol)
nonlinear = norm(f_full - f_lin)
@printf("  ‖f_nl(u) - K1*u‖ = %.4e  (nonlinear contribution at solution)\n",
        nonlinear)


# Stress comparison 
println("\n--- Stress comparison ---")

function stresses_std(grid, dh, cv, u, C)
    σ = zeros(3, getncells(grid))   # σ11, σ22, σ12
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        sc = zero(SymmetricTensor{2, 2})
        for qp in 1:getnquadpoints(cv)
            ε = function_symmetric_gradient(cv, qp, ue)
            sc += C ⊡ ε
        end
        sc /= getnquadpoints(cv)
        cid = cellid(cell)
        σ[1, cid] = sc[1, 1]
        σ[2, cid] = sc[2, 2]
        σ[3, cid] = sc[1, 2]
    end
    return σ
end

function stresses_pol(grid, dh, cv, u, mat)
    # Linearized SVK stress σ = λ tr(ε) I + 2μ ε. Using full Green-Lagrange
    # would add an O(‖∇u‖²) geometric-nonlinearity term that the linear-elastic
    # side does not contain, polluting the comparison.
    σ = zeros(3, getncells(grid))
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        s11 = 0.0; s22 = 0.0; s12 = 0.0
        for qp in 1:getnquadpoints(cv)
            ε = function_symmetric_gradient(cv, qp, ue)
            S = mat.λ * tr(ε) * one(ε) + 2 * mat.μ * ε
            s11 += S[1, 1]; s22 += S[2, 2]; s12 += S[1, 2]
        end
        nqp = getnquadpoints(cv)
        cid = cellid(cell)
        σ[1, cid] = s11 / nqp
        σ[2, cid] = s22 / nqp
        σ[3, cid] = s12 / nqp
    end
    return σ
end

σ_std = stresses_std(grid, dh, cv, u_std, C)
σ_pol = stresses_pol(grid, dh, cv, u_pol, mat)

abserr = maximum(abs.(σ_pol - σ_std))
relerr = abserr / maximum(abs.(σ_std))
@printf("  max|σ_pol - σ_std| = %.4e\n", abserr)
@printf("  Relative error      = %.4e\n", relerr)
println(relerr < 1e-8 ? "  ✓ Stresses match" : "  ✗ Stresses differ")
