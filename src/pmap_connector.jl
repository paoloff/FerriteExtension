# Generate .jl files with explicit polynomial for Poli-IMaP.
# The generated function uses only scalar +, -, * — GraphBuilder traces through it.
#
# Outputs:
#   filepath       → f_poly(q) = K1·q + K2(q⊗q) + K3(q⊗q⊗q)
#   If M provided  → also writes M as a dense matrix to a separate file (same dir, M_pmap.jl)

function generate_pmap_F(filepath::String,
                             K1::SparseMatrixCSC, K2::SparseT3, K3::SparseT4;
                             M::Union{Nothing, SparseMatrixCSC} = nothing,
                             tol::Float64 = 1e-15)

    n = size(K1, 1)

    # collect K1 COO entries grouped by output row
    Ir, Ic, Vk1 = findnz(K1)
    k1_by_row = [Tuple{Int,Float64}[] for _ in 1:n]
    for k in eachindex(Vk1)
        abs(Vk1[k]) > tol && push!(k1_by_row[Ir[k]], (Ic[k], Vk1[k]))
    end

    # collect K2 entries, summing duplicates with same (i,j,k)
    k2_acc = [Dict{Tuple{Int,Int}, Float64}() for _ in 1:n]
    for k in eachindex(K2.V)
        key = (K2.J[k], K2.K[k])
        d = k2_acc[K2.I[k]]
        d[key] = get(d, key, 0.0) + K2.V[k]
    end
    k2_by_row = [Tuple{Int,Int,Float64}[(j, k, v) for ((j, k), v) in sort(collect(d)) if abs(v) > tol]
                 for d in k2_acc]

    # collect K3 entries, summing duplicates with same (i,j,k,l)
    k3_acc = [Dict{Tuple{Int,Int,Int}, Float64}() for _ in 1:n]
    for k in eachindex(K3.V)
        key = (K3.J[k], K3.K[k], K3.L[k])
        d = k3_acc[K3.I[k]]
        d[key] = get(d, key, 0.0) + K3.V[k]
    end
    k3_by_row = [Tuple{Int,Int,Int,Float64}[(j, k, l, v) for ((j, k, l), v) in sort(collect(d)) if abs(v) > tol]
                 for d in k3_acc]

    # build expression string for f_i(q) for each row
    force_exprs = String[]
    for i in 1:n
        terms = String[]
        for (j, v) in k1_by_row[i]
            push!(terms, _fmt_term(v, "q[$j]"))
        end
        for (j, k, v) in k2_by_row[i]
            push!(terms, _fmt_term(v, "q[$j]*q[$k]"))
        end
        for (j, k, l, v) in k3_by_row[i]
            push!(terms, _fmt_term(v, "q[$j]*q[$k]*q[$l]"))
        end
        expr = isempty(terms) ? "zero(eltype(q))" : _join_terms(terms)
        push!(force_exprs, expr)
    end

    # ensure output directory exists
    mkpath(dirname(filepath))

    # write the file
    open(filepath, "w") do io
        println(io, "#= Auto-generated polynomial for Poli-IMaP")
        println(io, "   nDOF = $n")
        if !isnothing(M)
            println(io, "   Mode: first-order ODE   ẋ = F(x),  x = [q; q̇]")
            println(io, "   ddim = $(2n)")
        else
            println(io, "   Mode: force polynomial  f(q) = K1·q + K2(q⊗q) + K3(q⊗q⊗q)")
        end
        println(io, "=#\n")

        # always emit the force polynomial
        println(io, "# force polynomial: f(q) = K1·q + K2·(q⊗q) + K3·(q⊗q⊗q)")
        println(io, "function f_poly(q)")
        println(io, "    return [")
        for (i, expr) in enumerate(force_exprs)
            sep = i < n ? "," : ""
            println(io, "        $expr$sep")
        end
        println(io, "    ]")
        println(io, "end\n")

    end

    println("Generated Poli-IMaP force polynomial: $filepath  (nDOF = $n)")

    # export M as a dense matrix in a separate file
    if !isnothing(M)
        mpath = joinpath(dirname(filepath), "M_pmap.jl")
        Md = Matrix(M)
        open(mpath, "w") do io
            println(io, "#= Auto-generated mass matrix for Poli-IMaP")
            println(io, "   nDOF = $n")
            println(io, "   Usage: M_q̈ + f_poly(q) = 0")
            println(io, "=#\n")
            println(io, "const M_mass = [")
            for i in 1:n
                terms = String[]
                for j in 1:n
                    push!(terms, string(Md[i, j]))
                end
                sep = i < n ? ";" : ""
                println(io, "    ", join(terms, " "), sep)
            end
            println(io, "]")
        end
        println("Generated Poli-IMaP mass matrix:    $mpath  (nDOF = $n)")
    end
end


# format a single coefficient × monomial term
function _fmt_term(coeff::Float64, monomial::String)
    if coeff == 1.0
        return monomial
    elseif coeff == -1.0
        return "-$monomial"
    else
        return "$(coeff)*$monomial"
    end
end

# join terms with + / - signs
function _join_terms(terms::Vector{String})
    buf = IOBuffer()
    for (i, t) in enumerate(terms)
        if i == 1
            print(buf, t)
        elseif startswith(t, "-")
            print(buf, " - ", t[2:end])
        else
            print(buf, " + ", t)
        end
    end
    return String(take!(buf))
end
