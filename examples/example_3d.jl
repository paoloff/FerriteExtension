#= 
    example_3d.jl — 3D cantilever beam with hexahedral elements

    Brick domain meshed with Hexahedron elements, clamped on left face.
    Full 3D SVK material.

    Note: K3 extraction is expensive for 3D hex (24 DOFs/element → 24⁴
    entries per element). Use a coarse mesh, or higher-order elements with
    fewer cells, for tractability.
=#

include("../includes.jl")

# Parameters
Lx     = 100.0
Ly     = 10.0
Lz     = 10.0
nelx   = 5      # keep coarse — AD on 24-DOF elements is expensive
nely   = 1
nelz   = 1
E      = 210e3
ν      = 0.3
ρ      = 7.85e-9
outdir = "export_3d"

mat = SVKMaterial(; E=E, ν=ν, ρ=ρ)

# Mesh: 3D hexahedral elements
println("Setting up 3D model...")
grid = generate_grid(Hexahedron, (nelx, nely, nelz),
                     Vec((0.0, 0.0, 0.0)), Vec((Lx, Ly, Lz)))

ip = Lagrange{RefHexahedron, 1}()^3
qr = QuadratureRule{RefHexahedron}(2)
cv = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"),
                   (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]))
close!(ch)

nDOF = ndofs(dh)
@info "3D Model" nDOF nelems=getncells(grid)

# Extract geometry
println("Extracting element geometry...")
elems = extract_element_geoms(dh, cv)
println("  $(length(elems)) elements, sdim=$(elems[1].sdim)")
ndof_e = length(elems[1].gdofs)
println("  DOFs per element: $ndof_e")
@printf("  K3 entries per element: %d (= %d⁴)\n", ndof_e^4, ndof_e)

# Verify
verify_elem_forces(dh, cv, elems, mat)

# Assemble
println("\n=== Assembling mass matrix ===")
M = assemble_mass(elems, nDOF, ρ)
@printf("  M: %d×%d, nnz=%d\n", nDOF, nDOF, nnz(M))

println("\n=== Extracting K1, K2, K3 via ForwardDiff ===")
println("  (this may take a while for 3D elements...)")
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
meta = Dict("nDOF" => nDOF, "dimension" => 3,
            "nElements" => getncells(grid), "method" => "ForwardDiff_exact_AD",
            "geometry" => Dict("Lx" => Lx, "Ly" => Ly, "Lz" => Lz))
export_tensors(outdir, M, K1, K2, K3; meta=meta)

println("\n=== Done (3D) ===")
