#= 
    example_1d.jl — 1D nonlinear bar (axial deformation)

    Bar of length L, clamped at the left end, meshed with Line elements.
    SVK material in 1D: σ = (λ+2μ)ε + finite-strain nonlinear terms.

 =#

include("../includes.jl")

# Parameters
L      = 1. #100.0
nelx   = 20
E      = 1. #210e3
ν      = 0.0       # 1D: irrelevant, but needed for the constructor
ρ      = 1. #7.85e-9
outdir = "examples/exports"

mat = SVKMaterial(; E=E, ν=ν, ρ=ρ)

# Mesh: 1D line elements
println("Setting up 1D bar model...")
grid = generate_grid(Line, (nelx,), Vec((0.0,)), Vec((L,)))

ip = Lagrange{RefLine, 1}()^1    # 1-component vector field
qr = QuadratureRule{RefLine}(2)
cv = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> (0.0,), [1]))
close!(ch)

nDOF = ndofs(dh)
@info "1D Model" nDOF nelx

# Extract geometry
println("Extracting element geometry...")
elems = extract_element_geoms(dh, cv)
println("  $(length(elems)) elements, sdim=$(elems[1].sdim)")

# Verify
verify_elem_forces(dh, cv, elems, mat)

# Assemble
println("\n=== Assembling mass matrix ===")
M = assemble_mass(elems, nDOF, ρ)
@printf("  M: %d×%d, nnz=%d\n", nDOF, nDOF, nnz(M))

println("\n=== Extracting K1, K2, K3 via ForwardDiff ===")
K1, K2, K3 = assemble_poly_tensors(elems, mat, nDOF)
@printf("  K1: nnz=%d\n", nnz(K1))
@printf("  K2: %d entries\n", length(K2.V))
@printf("  K3: %d entries\n", length(K3.V))

# Verify polynomial
println()
verify_poly_force(elems, mat, K1, K2, K3, nDOF)

# Apply BCs
println("\n=== Applying boundary conditions ===")
cdofs  = apply_dirichlet!(K1, M, ch)
K2, K3 = apply_dirichlet(K2, K3, cdofs)
@printf("  K2 after BC filter: %d entries\n", length(K2.V))
@printf("  K3 after BC filter: %d entries\n", length(K3.V))

# Export
println("\n=== Exporting ===")
meta = Dict("nDOF" => nDOF, "dimension" => 1,
            "nElements" => nelx, "method" => "ForwardDiff_exact_AD")
export_tensors(outdir, M, K1, K2, K3; meta=meta)

# Generate Poli-IMaP function file
println("\n=== Generating Poli-IMaP F(x) ===")
generate_pmap_F(joinpath(outdir, "F_pmap_1d.jl"), K1, K2, K3; M=M)

println("\n=== Done (1D) ===")
