# SVKMaterial — Saint-Venant Kirchhoff hyperelastic material.
# Holds the two Lamé parameters and the mass density.

struct SVKMaterial
    λ::Float64    # 1st Lamé parameter
    μ::Float64    # 2nd Lamé parameter (shear modulus)
    ρ::Float64    # density
end

# Build SVK material from engineering constants (E, ν, ρ).
# `plane_stress=true` selects the 2D plane-stress λ; otherwise the 3D / 1D form.
function SVKMaterial(; E=210e3, ν=0.3, ρ=7.85e-9, plane_stress=false)
    μ = E / (2(1 + ν))
    λ = plane_stress ? E * ν / ((1 + ν) * (1 - ν)) :
                       E * ν / ((1 + ν) * (1 - 2ν))
    return SVKMaterial(λ, μ, ρ)
end
