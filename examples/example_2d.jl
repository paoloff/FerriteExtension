#= 
    example_2d.jl — 2D plane-stress cantilever beam

    Rectangular domain meshed with Quadrilateral elements, clamped on left.
    SVK material with plane-stress Lamé parameters.
=#

include("../includes.jl")

# Parameters
Lx     = 1.  # 100.0
Ly     = .1 # 10.0
nelx   = 10
nely   = 2
E      = 1. #210e3
ν      = 0.3
ρ      = 1. #7.85e-9
outdir = "examples/exports"

mat = SVKMaterial(; E=E, ν=ν, ρ=ρ, plane_stress=true)

# Mesh: 2D quadrilateral elements
println("Setting up 2D plane-stress model...")
grid = generate_grid(Quadrilateral, (nelx, nely),
                     Vec((0.0, 0.0)), Vec((Lx, Ly)))

ip = Lagrange{RefQuadrilateral, 1}()^2
qr = QuadratureRule{RefQuadrilateral}(2)
cv = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"),
                   (x, t) -> (0.0, 0.0), [1, 2]))
close!(ch)

nDOF = ndofs(dh)
@info "2D Model" nDOF nelems=getncells(grid)

# Extract geometry
println("Extracting element geometry...")
elems = extract_element_geoms(dh, cv);
println("  $(length(elems)) elements, sdim=$(elems[1].sdim)")

# Verify
verify_elem_forces(dh, cv, elems, mat)

# Assemble
println("\n=== Assembling mass matrix ===")
M = assemble_mass(elems, nDOF, ρ)
@printf("  M: %d×%d, nnz=%d\n", nDOF, nDOF, nnz(M))

println("\n=== Extracting K1, K2, K3 via ForwardDiff ===")
K1, K2, K3 = assemble_poly_tensors(elems, mat, nDOF);
@printf("  K1: nnz=%d\n", nnz(K1))
@printf("  K2: %d entries\n", length(K2.V))
@printf("  K3: %d entries\n", length(K3.V))

# Verify polynomial
println()
verify_poly_force(elems, mat, K1, K2, K3, nDOF)

# Apply BCs
println("\n=== Applying boundary conditions ===")
cdofs  = apply_dirichlet!(K1, M, ch)
K2, K3 = apply_dirichlet(K2, K3, cdofs);
@printf("  K2 after BC filter: %d entries\n", length(K2.V))
@printf("  K3 after BC filter: %d entries\n", length(K3.V))

# Export
println("\n=== Exporting ===")
meta = Dict("nDOF" => nDOF, "dimension" => 2,
            "nElements" => getncells(grid), "method" => "ForwardDiff_exact_AD",
            "geometry" => Dict("Lx" => Lx, "Ly" => Ly))
export_tensors(outdir, M, K1, K2, K3; meta=meta)

# Generate Poli-IMaP function file
println("\n=== Generating Poli-IMaP F(x) ===")
generate_pmap_F(joinpath(outdir, "F_pmap_2d.jl"), K1, K2, K3; M=M)

println("\n=== Done (2D) ===")
